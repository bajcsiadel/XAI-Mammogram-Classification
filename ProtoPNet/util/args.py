import argparse
import json
import os
import pandas as pd
import pickle

from pathlib import Path

from ProtoPNet.dataset.metadata import DATASETS, DataFilter
from ProtoPNet.config.backbone_features import BACKBONE_MODELS

from ProtoPNet.util import helpers
from ProtoPNet.util import errors


def get_args():
    """
    Parse the arguments given to the program through the command line
    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser("Train a ProtoPNet")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=BACKBONE_MODELS,
        default="resnet18",
        help="Backbone used to train the network",
    )
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="If set, only train the backbone without prototype layer",
    )
    parser.add_argument(
        "--add-on-layers-type",
        type=str,
        default="regular",
        choices=["regular", "bottleneck"],
        help="The type of add-on layers to use",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="If set, use pretrained weights for the backbone",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS.keys(),
        help="Dataset used to train the network",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Name of the column from the csv containing the target labels"
    )
    parser.add_argument(
        "--masked",
        action="store_true",
        help="If set, use masked data",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="If set, use preprocessed data",
    )
    parser.add_argument(
        "--only-MLO",
        action=__FilterMLO,
        dest="data_filters",
        default=[],
        nargs=0,
        help="If set, only use MLO views",
    )
    parser.add_argument(
        "--only-CC",
        action=__FilterCC,
        dest="data_filters",
        default=[],
        nargs=0,
        help="If set, only use CC views",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="If set, use data augmentation during training",
    )
    parser.add_argument(
        "--cross-validation-folds",
        type=int,
        default=5,
        help="Number of folds for cross validation",
    )
    parser.add_argument(
        "--stratified-cross-validation",
        action="store_true",
        help="Use stratified cross-validation"
    )
    parser.add_argument(
        "--grouped-cross-validation",
        action="store_true",
        help="Use grouped cross-validation"
    )
    parser.add_argument(
        "--prototypes-per-class",
        type=int,
        default=10,
        help="Number of prototypes per class",
    )
    parser.add_argument(
        "--prototype-size",
        type=int,
        default=256,
        help="Size of the prototype vectors",
    )
    parser.add_argument(
        "--prototype-activation-function",
        type=str,
        default="log",
        choices=["log", "linear", "relu", "sigmoid", "tanh"],
        help="Activation function used for the prototypes",
    )
    # warm optimizer learning rates
    parser.add_argument(
        "--warm-lr-add-on-layers",
        type=float,
        default=3e-3,
        help="Learning rate for the add-on layers in the warm optimizer",
    )
    parser.add_argument(
        "--warm-lr-prototype-vectors",
        type=float,
        default=3e-3,
        help="Learning rate for the prototype vectors in the warm optimizer",
    )
    # ---------------------------------------------
    # joint optimizer learning rates
    parser.add_argument(
        "--joint-lr-features",
        type=float,
        default=1e-4,
        help="Learning rate for the features in the joint optimizer",
    )
    parser.add_argument(
        "--joint-lr-add-on-layers",
        type=float,
        default=3e-3,
        help="Learning rate for the add-on layers in the joint optimizer",
    )
    parser.add_argument(
        "--joint-lr-prototype-vectors",
        type=float,
        default=3e-3,
        help="Learning rate for the prototype vectors in the joint optimizer",
    )
    parser.add_argument(
        "--joint-lr-step-size",
        type=int,
        default=5,
        help="Step size for the learning rate scheduler of the joint optimizer",
    )
    # ---------------------------------------------
    parser.add_argument(
        "--last-layer-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the last layer optimizer",
    )
    # ---------------------------------------------
    # coefficients
    parser.add_argument(
        "--cross-entropy-coefficient",
        type=float,
        default=1,
        help="Coefficient for the cross entropy loss",
    )
    parser.add_argument(
        "--clustering-coefficient",
        type=float,
        default=8e-1,
        help="Coefficient for the clustering loss",
    )
    parser.add_argument(
        "--separation-coefficient",
        type=float,
        default=6e-1,
        help="Coefficient for the separation loss",
    )
    parser.add_argument(
        "--separation-margin-coefficient",
        type=float,
        default=1.0,
        help="Coefficient for the separation margin loss",
    )
    parser.add_argument(
        "--l1-coefficient",
        type=float,
        default=1e-4,
        help="Coefficient for the L1 loss",
    )
    parser.add_argument(
        "--l2-coefficient",
        type=float,
        default=1e-2,
        help="Coefficient for the L2 loss",
    )
    # ---------------------------------------------
    parser.add_argument(
        "--binary-cross-entropy",
        action="store_true",
        help="If set, use binary cross entropy loss instead of cross entropy loss",
    )
    parser.add_argument(
        "--separation-type",
        type=str,
        choices=["max", "avg", "margin"],
        default="avg",
        help="The type of separation loss to use",
    )
    parser.add_argument(
        "--number-of-workers",
        type=int,
        default=4,
        help="Number of workers used to load the data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        action=__PowerOfTwo,
        help="Batch size used to train the network",
    )
    parser.add_argument(
        "--batch-size-pretrain",
        type=int,
        default=128,
        action=__PowerOfTwo,
        help="Batch size when pretraining the prototypes (first training stage)",
    )
    parser.add_argument(
        "--batch-size-push",
        type=int,
        default=64,
        action=__PowerOfTwo,
        help="Batch size when pushing back the prototypes"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="The number of epochs ProtoPNet should be trained (second training stage)",
    )
    parser.add_argument(
        "--epochs-pretrain",
        type=int,
        default=10,
        help="Number of epochs to pre-train the prototypes (first training stage). "
             "Recommended to train at least until the align loss < 1",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=10,
        help="During fine tuning, only train classification layer and freeze rest. "
             "Usually done for a few epochs (at least 1, more depends on size of dataset)",
    )
    parser.add_argument(
        "--push-start",
        type=int,
        default=2,
        help="Epoch from which prototypes should be pushed to the database",
    )
    parser.add_argument(
        "--push-intervals",
        type=int,
        default=5,
        help="Difference between epochs in which prototypes should be "
             "pushed to the database",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="The id of the GPU that should be used to train ProtoPNet",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="The directory in which train progress should be logged",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed used for random number generators",
    )
    parser.add_argument(
        "--args",
        type=str,
        help="The path to a file containing the arguments to be used for training",
    )

    args = parser.parse_args()

    __process_agrs(args)

    return args


def save_args(args, location):
    """
    Save the arguments in the specified directory as
        - a json file called 'args.json'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :type args: argparse.Namespace
    :param location: The path to the directory where the arguments should be saved
    :type location: Path
    """
    # If the specified directory does not exist, create it
    location.mkdir(parents=True, exist_ok=True)
    # Save the args in a json file
    with (location / "args.json").open(mode="w") as fd:
        json.dump(vars(args), fd, indent=4, cls=helpers.EnhancedJSONEncoder)
    # Pickle the args for possible reuse
    with (location / "args.pickle").open(mode="wb") as fd:
        pickle.dump(args, fd)


def __process_agrs(args):
    args.dataset_config = DATASETS[args.dataset]
    args.used_images = __get_used_images_key(args)

    helpers.set_used_images(args.dataset_config, args.used_images, args.target)

    if not args.augment:
        args.dataset_config.IMAGE_PROPERTIES.AUGMENTATIONS.DISABLED = True

    dataset_meta = pd.read_csv(
        args.dataset_config.METADATA.FILE,
        **args.dataset_config.METADATA.PARAMETERS,
    )

    args.classes = dataset_meta[(args.target, "label")][~dataset_meta[(args.target, "label")].isna()].unique().tolist()
    args.number_of_classes = len(args.classes)

    args.prototype_shape = (args.prototypes_per_class * args.number_of_classes, args.prototype_size, 1, 1)

    args.push_epochs = [i + args.push_start for i in range(args.epochs) if i % args.push_intervals == 0]
    # add push to last epoch
    args.push_epochs = args.push_epochs + [args.epochs + 1]

    args.loss_coefficients = {
        "cross_entropy": args.cross_entropy_coefficient,
        "clustering": args.clustering_coefficient,
        "separation": args.separation_coefficient,
        "separation_margin": args.separation_margin_coefficient,
        "l1": args.l1_coefficient,
        "l2": args.l2_coefficient,
    }

    if args.log_dir == "":
        args.log_dir = f"{args.dataset}-{args.used_images}-{args.backbone}-{args.target}"
    args.log_dir = Path(os.getenv("PROJECT_ROOT")) / "runs" / args.log_dir


def __get_used_images_key(args):
    if args.masked and args.preprocessed:
        return "masked_preprocessed"
    elif args.masked:
        return "masked"
    else:
        return "original"


def generate_gin_config(args, location):
    """
    Generate a gin config file from the given arguments
    :param args: The arguments to generate the gin config file from
    :type args: argparse.Namespace
    :param location: The path to the directory where the gin config file should be saved
    :type location: Path
    :returns: name of the config file
    :rtype: Path
    """
    match args.dataset.upper():
        case "MIAS":
            data_module = "MIASDataModule"
        case "DDSM":
            data_module = "DDSMDataModule"
        case _:
            raise ValueError(f"Unknown dataset ({args.dataset})")

    constants_config_file = location / "constants.gin"

    constants_config_content = f"# IMAGE PROPERTIES\n"
    constants_config_content += f"const_image_mean     = {args.dataset_config.USED_IMAGES.MEAN}\n"
    constants_config_content += f"const_image_std      = {args.dataset_config.USED_IMAGES.STD}\n"
    constants_config_content += f"const_image_max      = {args.dataset_config.IMAGE_PROPERTIES.MAX_VALUE}\n"
    constants_config_content += f"const_image_shape    = {args.dataset_config.IMAGE_PROPERTIES.SHAPE}\n"
    constants_config_content += f"const_image_channels = {args.dataset_config.IMAGE_PROPERTIES.COLOR_CHANNELS}\n"
    constants_config_content += f"\n"
    constants_config_content += f"# DATASET PROPERTIES\n"
    constants_config_content += f"const_dataset_number_of_classes = {args.number_of_classes}\n"
    constants_config_content += f"\n"
    constants_config_content += f"# PROTOTYPE PROPERTIES\n"
    constants_config_content += f"const_prototype_number_per_class = {args.prototypes_per_class}\n"
    constants_config_content += f"const_prototype_shape            = {args.prototype_shape}\n"
    constants_config_content += f"\n"
    constants_config_content += f"# PROTOPNET PROPERTIES\n"
    constants_config_content += f"const_protopnet_separation_type = '{args.separation_type}'\n"
    constants_config_content += f"const_protopnet_backbone_only   = {args.backbone_only}\n"

    constants_config_file.write_text(constants_config_content)

    train_config_file = location / "train.gin"

    train_config_content = "# include constants\n"
    train_config_content += f"include '{constants_config_file}'\n"
    train_config_content += f"\n"
    for fn_name in ["train", "test"]:
        train_config_content += f"{fn_name}.prototype_shape   = %const_prototype_shape\n"
        train_config_content += f"{fn_name}.separation_type   = %const_protopnet_separation_type\n"
        train_config_content += f"{fn_name}.number_of_classes = %const_dataset_number_of_classes\n"
        train_config_content += f"{fn_name}.class_specific    = True\n"
        train_config_content += f"{fn_name}.backbone_only     = %const_protopnet_backbone_only\n"
        train_config_content += f"{fn_name}.use_bce           = {args.binary_cross_entropy}\n"
        train_config_content += f"{fn_name}.loss_coefficients = {json.dumps(args.loss_coefficients)}\n"
        train_config_content += f"\n"

    train_config_file.write_text(train_config_content)

    config_file = location / "config.gin"

    # Generate the gin config
    config_content = "# include constants\n"
    config_content += f"include '{constants_config_file}'\n"
    config_content += "\n"
    config_content += "# include train parameters\n"
    config_content += f"include '{train_config_file}'\n"
    config_content += "\n"
    config_content += f"{data_module}.used_images            = '{args.used_images}'\n"
    config_content += f"{data_module}.classification         = '{args.target}'\n"
    data_filters = ",\n\t".join(map(lambda data_filter: f"@{data_filter.SCOPE}()", args.data_filters))
    if len(args.data_filters) > 0:
        data_filters = f"\n\t{data_filters},\n"
    config_content += f"{data_module}.data_filters           = [{data_filters}]\n"
    config_content += f"{data_module}.cross_validation_folds = {args.cross_validation_folds}\n"
    config_content += f"{data_module}.stratified             = {args.stratified_cross_validation}\n"
    config_content += f"{data_module}.groups                 = {args.grouped_cross_validation}\n"
    config_content += f"{data_module}.num_workers            = {args.number_of_workers}\n"
    config_content += f"{data_module}.seed                   = {args.seed}\n"
    config_content += f"\n"

    for data_filter in args.data_filters:
        config_content += f"{data_filter.SCOPE}.FIELD = {data_filter.FIELD}\n"
        if type(data_filter.VALUE) is str:
            value = f"'{data_filter.VALUE}'"
        else:
            value = str(data_filter.VALUE)
        config_content += f"{data_filter.SCOPE}.VALUE = {value}\n"
        config_content += f"\n"

    config_content += f"preprocess.mean               = %const_image_mean\n"
    config_content += f"preprocess.std                = %const_image_std\n"
    config_content += f"preprocess.number_of_channels = %const_image_channels\n"
    config_content += f"\n"
    config_content += f"undo_preprocess.mean               = %const_image_mean\n"
    config_content += f"undo_preprocess.std                = %const_image_std\n"
    config_content += f"undo_preprocess.number_of_channels = %const_image_channels\n"
    config_content += f"\n"
    config_content += f"ResNet_features.color_channels = %const_image_channels\n"
    config_content += f"\n"
    config_content += f"construct_PPNet.base_architecture              = '{args.backbone}'\n"
    config_content += f"construct_PPNet.pretrained                     = {args.pretrained}\n"
    config_content += f"construct_PPNet.img_shape                      = %const_image_shape\n"
    config_content += f"construct_PPNet.prototype_shape                = %const_prototype_shape\n"
    config_content += f"construct_PPNet.num_classes                    = {args.number_of_classes}\n"
    config_content += f"construct_PPNet.prototype_activation_function  = '{args.prototype_activation_function}'\n"
    config_content += f"construct_PPNet.add_on_layers_type             = '{args.add_on_layers_type}'\n"
    config_content += f"construct_PPNet.backbone_only                  = %const_protopnet_backbone_only\n"
    config_content += f"construct_PPNet.positive_weights_in_classifier = False\n"
    config_content += f"\n"
    config_content += f"main.dataset_module = @{data_module}()\n"

    config_file.write_text(config_content)

    return config_file


def load_args(arg_file):
    """
    Load the arguments from the specified file
    :param arg_file: The path to the file where the arguments should be loaded from
    :type arg_file: Path
    :raises errors.UnsupportedExtensionError: If the file extension is not supported
    :return: argparse.Namespace
    """
    _, file_ext = arg_file.suffix
    if file_ext == ".json":
        # Load the args from a json file
        with arg_file.open(mode="r") as fd:
            args = json.load(fd)
        args = argparse.Namespace(**args)
    elif file_ext == ".pickle":
        # Load the args from a pickle file
        with arg_file.open(mode="rb") as fd:
            args = pickle.load(fd)
    else:
        raise errors.UnsupportedExtensionError(f"Unknown file extension ({file_ext}) to load arguments from!\n"
                                               f"Use a .json or .pickle file instead.")
    return args


class __PowerOfTwo(argparse.Action):
    @staticmethod
    def __is_power_of_two(number):
        return number != 0 and ((number & (number - 1)) == 0)

    def __call__(self, parser, namespace, values, option_string=None):
        if not self.__is_power_of_two(values):
            raise argparse.ArgumentError(self, "Given number should be a power of two!")
        setattr(namespace, self.dest, values)


class __AppendFilter(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        filters = getattr(namespace, self.dest)
        filters.append(values)
        setattr(namespace, self.dest, filters)


class __FilterView(__AppendFilter):
    def __call__(self, parser, namespace, view, option_string=None):
        super().__call__(parser, namespace, DataFilter(FIELD=("mammogram_properties", "view"), VALUE=view),
                         option_string)


class __FilterMLO(__FilterView):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, "MLO", option_string)


class __FilterCC(__FilterView):
    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, "CC", option_string)
