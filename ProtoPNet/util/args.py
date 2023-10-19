import argparse
import json
import numpy as np
import os
import pickle

from ProtoPNet.config.datasets import DATASETS
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
        "--prototypes-per-classes",
        type=int,
        default=10,
        help="Number of prototypes per class",
    )
    parser.add_argument(
        "--prototype-activation-function",
        type=str,
        default="log",
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
        default="max",
        help="The type of separation loss to use",
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
        help="Batch size when pretraining the prototypes (first training stage)",
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
        "--push-epochs",
        type=int,
        help="Differnece between epochs in which prototypes should be "
             "pushed to the database",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="The id of the GPU that should be used to train ProtoPNet",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs/train_protopnet",
        help="The directory in which train progress should be logged",
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
    :type location: str
    """
    # If the specified directory does not exist, create it
    helpers.makedir(location)
    # Save the args in a json file
    with open(os.path.join(location, "args.json"), "w") as fd:
        json.dump(vars(args), fd)
    # Pickle the args for possible reuse
    with open(os.path.join(location, "args.pickle"), "wb") as fd:
        pickle.dump(args, fd)


def __process_agrs(args):
    dataset_config = DATASETS[args.dataset]
    args.img_shape = dataset_config["IMAGE_SHAPE"]
    args.channels = dataset_config["COLOR_CHANNELS"]

    args.classes = dataset_config["CLASSES"]
    args.number_of_classes = dataset_config["NUMBER_OF_CLASSES"]
    assert args.classes != [] and args.number_of_classes == len(args.classes), \
        "Number of classes does not match length of classes list"

    args.prototype_shape = (args.prototypes_per_classes * args.number_of_classes, 256, 1, 1)

    args.push_epochs = [ i for i in np.arange(args.epochs) + args.push_start if i % args.push_epochs == 0 ]


def load_args(arg_file):
    """
    Load the arguments from the specified file
    :param arg_file: The path to the file where the arguments should be loaded from
    :type arg_file: str
    :raises errors.UnsupportedExtensionError: If the file extension is not supported
    :return: argparse.Namespace
    """
    _, file_ext = os.path.splitext(arg_file)
    if file_ext == ".json":
        # Load the args from a json file
        with open(os.path.join(arg_file, "args.json"), "r") as fd:
            args = json.load(fd)
        args = argparse.Namespace(**args)
    elif file_ext == ".pickle":
        # Load the args from a pickle file
        with open(os.path.join(arg_file, "args.pickle"), "rb") as fd:
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
            raise argparse.ArgumentError("Given number should be a power of two!")
        setattr(namespace, self.dest, values)