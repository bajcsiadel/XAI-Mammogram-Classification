"""
Visualizes the activation of prototypes per each class on given image/images.
The loaded model must be after push ({fold}-{epoch}-push-{accu}.pth).
"""
import dataclasses as dc
import copy
import json
import re
from collections import namedtuple
from pathlib import Path

import albumentations as A
import cv2
import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.utils.data
from albumentations.pytorch import ToTensorV2
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.autograd import Variable

from xai_mam.models.ProtoPNet import construct_model
from xai_mam.models.ProtoPNet._helpers import find_high_activation_crop
from xai_mam.utils.config.resolvers import resolve_run_location, resolve_create
from xai_mam.utils.config.types import Gpu, Outputs as ModelOutputs
from xai_mam.utils.environment import get_env
from xai_mam.utils.errors.file_errors import CsvMismatchedColumnsError
from xai_mam.utils.log import ScriptLogger, TrainLogger


@dc.dataclass
class Dirs:
    missed: Path
    correct: Path
    # defined from the code
    checkpoints: Path = Path("None")
    model: Path = Path("None")
    saved_images: Path = Path("None")

    def __iter__(self):
        return iter(list(filter(lambda item: item[1] != "None", self.__dict__.items())))


@dc.dataclass
class FilePrefixes:
    heatmap: str
    max_activation_patch: str
    max_activation_patch_heatmap: str
    max_activation_bounding_box: str


@dc.dataclass
class Save:
    heatmaps: bool
    max_activation_patch: bool
    max_activation_patch_heatmap: bool
    max_activation_bounding_box: bool


@dc.dataclass
class Outputs:
    dirs: Dirs
    file_prefixes: FilePrefixes
    to_save: Save


@dc.dataclass
class ScriptConfig:
    result_dir: Path
    model_path: Path
    outputs: Outputs
    class_specific: bool
    image_list_file: Path = Path("None")
    image_path: Path = Path("None")
    image_class: int = -1
    gpu: Gpu = dc.field(default_factory=Gpu)

    def __post_init__(self):
        if self.image_list_file == Path("None") and self.image_path == Path("None"):
            raise AttributeError(
                "Either 'example_image_file' or 'image_path' must be provided."
            )
        if not self.image_list_file == Path("None"):
            if (not self.image_list_file.is_file()
                    or self.image_list_file.suffix != ".csv"):
                raise AttributeError("'example_image_file' must be a valid csv file.")
        else:
            # image path is given
            if not self.image_path.is_file():
                raise FileNotFoundError("Image not found at the given path.")
            elif self.image_class < 0:
                raise AttributeError("Image class must be provided.")

    Image = namedtuple("Image", ["path", "true_class"])

    def __read_image_date(self) -> list[Image]:
        """
        Read the image data from the given file path.
        The file must contain two columns: 'path' and 'class'.

        :return: the read data
        :raises CsvMismatchedColumnsError: if the file does not contain the required columns
        """
        data = pd.read_csv(self.image_list_file, header=0)
        if set(self.Image._fields) < set(data.columns):
            raise CsvMismatchedColumnsError(
                "Required column ('path' or 'true_class') is missing!"
            )
        data["path"] = data["path"].apply(Path)
        return list(map(lambda image: self.Image(**image), data.T.to_dict().values()))

    def get_images(self) -> list[Image]:
        if self.image_list_file != Path("None"):
            return self.__read_image_date()
        return [self.Image(**{
            "path": Path(self.image_path),
            "true_class": self.image_class
        })]


def create_dirs(cfg: ScriptConfig, logger: ScriptLogger) -> Dirs:
    """
    Create directories for the outputs.

    :param cfg:
    :param logger:
    :return:
    """
    checkpoints_dir = cfg.model_path.parent
    logger.info(f"Checkpoints dir: {checkpoints_dir}")

    model_name = cfg.model_path.stem
    logger.info(f"Model name: {model_name}")

    model_dir = checkpoints_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "checkpoints": checkpoints_dir,
        "model": model_dir
    }

    for dir_key, dir_name in cfg.outputs.dirs:
        if dir_name != Path("None"):
            dir_path = model_dir / dir_name
            dirs[dir_key] = dir_path
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)

    return Dirs(**dirs)


def get_prototype_information(
        model_name: str,
        dirs: Dirs,
        model_train_parameters: omegaconf.DictConfig,
        logger: ScriptLogger
) -> tuple[int | None, int, np.ndarray]:
    """
    Get prototype information from the given model.

    :param model_name: name of the model
    :param dirs:
    :param model_train_parameters: parameters used for training the model
    :param logger:
    :returns:
    :raises FileNotFoundError: if the prototype information is not found
    """
    # define fold
    fold = re.search(r"^(\d)-", model_name)
    if fold is not None:
        fold = fold.group(1)

    # define epoch
    if fold is None:
        epoch = re.search(r"^(\d+)-", model_name).group(1)
    else:
        epoch = re.search(r"-(\d+)-", model_name).group(1)
    epoch = int(epoch)

    dirs.saved_images = (dirs.checkpoints.parent /
                         model_train_parameters.outputs.dirs.image)

    trys = []

    def find_closest_push(epoch_: int) -> int | None:
        logger.warning(
            f"Files {trys} were not found. "
            f"Searching for the closest (but lower) push epoch."
        )
        dir_name_template = "epoch" if fold is None else f"{fold}-epoch"
        while epoch_ > 0:
            if (d := (dirs.saved_images / f"{dir_name_template}-{epoch_}")).is_dir():
                logger.warning(f"Using prototypes from {d}")
                return epoch_
            epoch_ -= 1
        raise FileNotFoundError(f"Prototypes not found!")

    modify_epoch_functions = [
        lambda e: e,
        lambda e: e - model_train_parameters.model.phases["warm"].epochs,
        find_closest_push,
    ]

    for modify_epoch_fn in modify_epoch_functions:
        new_epoch = modify_epoch_fn(epoch)
        prototype_info_file = (
                dirs.saved_images /
                f"{fold}-epoch-{new_epoch}" /
                f"{model_train_parameters.outputs.file_prefixes.bounding_box}"
                f"-{new_epoch}.npy"
        )

        if prototype_info_file.is_file():
            prototype_info = np.load(prototype_info_file)
            dirs.saved_images = dirs.saved_images / f"{fold}-epoch-{new_epoch}"
            logger.debug(f"Prototype info read from {prototype_info_file}")
            return fold, epoch, prototype_info
        else:
            trys.append(prototype_info_file)


def read_image(image_path: Path) -> np.ndarray:
    """
    Read the image from the given path.

    :param image_path: location of the image
    :return: the content of the image
    """
    if image_path.suffix in [".npy", ".npz"]:
        return np.load(image_path)["image"]
    else:
        return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)


def save_highest_activation_patch(
        image_output: Path,
        image_name: str,
        logger: ScriptLogger,
        max_activation_pattern: np.ndarray,
        test_image: np.ndarray,
):
    highest_activation_patch_indices = find_high_activation_crop(
        max_activation_pattern
    )
    highest_activation_patch = test_image[
        highest_activation_patch_indices[0]:highest_activation_patch_indices[1],
        highest_activation_patch_indices[2]:highest_activation_patch_indices[3],
    ]
    logger.save_image(
        image_name,
        highest_activation_patch,
        image_output
    )


def save_highest_activation_bounding_box(
        image_output: Path,
        image_name: str,
        logger: ScriptLogger,
        max_activation_pattern: np.ndarray,
        test_image: np.ndarray
):
    highest_activation_patch_indices = find_high_activation_crop(
        max_activation_pattern
    )
    if len(test_image.shape) < 3:
        test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

    with_bounding_box = copy.deepcopy(test_image)
    with_bounding_box = cv2.rectangle(
        with_bounding_box,
        (highest_activation_patch_indices[2], highest_activation_patch_indices[0]),
        (highest_activation_patch_indices[3], highest_activation_patch_indices[1]),
        color=(255, 0, 0),
        thickness=1,
    )
    logger.save_image(
        image_name,
        with_bounding_box,
        image_output
    )


def get_heatmap(max_activation_pattern: np.ndarray) -> np.ndarray:
    """
    Generate the heatmap based on maximum activation of prototypes.

    :param max_activation_pattern:
    :return: heatmap
    """
    rescaled_activation_patter = (max_activation_pattern
                                  - np.amin(max_activation_pattern))
    rescaled_activation_patter = (rescaled_activation_patter
                                  / np.amax(rescaled_activation_patter))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * rescaled_activation_patter), cv2.COLORMAP_JET,
    )
    return heatmap[..., ::-1]


def get_maximum_activation_pattern_of_class_prototypes(
        class_index: int,
        image_shape: tuple[int, ...],
        logger: ScriptLogger,
        model: torch.nn.Module,
        n_prototypes_per_class: int,
        predicted: int,
        prototype_activation_patterns: torch.Tensor,
        prototype_activations: torch.Tensor,
        prototype_img_identity: np.ndarray,
        prototype_max_connection: np.ndarray,
        meta_file: pd.DataFrame,
):
    max_activation_pattern = np.zeros(image_shape)
    image_name = meta_file.index[-1]
    for prototype_index in range(n_prototypes_per_class):
        flat_prototype_index = class_index * n_prototypes_per_class + prototype_index

        if (prototype_max_connection[flat_prototype_index]
                != prototype_img_identity[flat_prototype_index]):
            prototype_connection_value = prototype_max_connection[
                flat_prototype_index]
            logger.debug(
                f"Prototype connection identity: {prototype_connection_value}")

        activation_value = prototype_activations[0][flat_prototype_index].item()
        weight = model.last_layer.weight[predicted][flat_prototype_index].item()

        meta_file.at[
            image_name,
            f"class-{class_index}-prototype-{prototype_index}-activation"
        ] = activation_value
        meta_file.at[
            image_name,
            f"class-{class_index}-prototype-{prototype_index}-weight"
        ] = weight

        activation_pattern = (
                prototype_activation_patterns[0][flat_prototype_index]
                .detach().cpu().numpy()
                * model.last_layer.weight[predicted][flat_prototype_index]
                .detach().cpu().numpy()
        )
        upsampled_activation_pattern = cv2.resize(
            activation_pattern,
            dsize=image_shape,
            interpolation=cv2.INTER_CUBIC,
        )
        max_activation_pattern = np.maximum(
            upsampled_activation_pattern, max_activation_pattern
        )

        with logger.increase_indent_context():
            logger.debug(f"Prototype index: {flat_prototype_index}")
            logger.debug(
                f"Prototype class identity: "
                f"{prototype_img_identity[flat_prototype_index]}"
            )
            logger.debug(
                f"Activation value (similarity score): {activation_value}"
            )
            logger.debug(f"Last layer weight with predicted class: {weight}")
    return max_activation_pattern


def initialize_meta_file() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["image_path", "expected", "predicted"],
        index=pd.Index([], name="image_name")
    )


def run(cfg: ScriptConfig, logger: ScriptLogger):
    """
    Generate activations for the given image.

    :return:
    """
    dirs = create_dirs(cfg, logger)

    images = cfg.get_images()
    logger.info(f"Number of images: {len(images)}")
    logger.debug("Images:")
    for image in images:
        logger.debug(f"\t{image}")

    model_train_config_file = dirs.checkpoints.parent / "conf" / "config.yaml"
    model_train_parameters = OmegaConf.load(model_train_config_file)
    logger.debug(
        f"Model train parameters: "
        f"{json.dumps(OmegaConf.to_container(model_train_parameters), indent=4)}"
    )

    saved_states = torch.load(cfg.model_path, map_location="cpu")
    model = construct_model(
        **saved_states["model_initialization_parameters"],
        logger=TrainLogger(
            f"{__name__}.model",
            ModelOutputs(**model_train_parameters.outputs)),
    )
    model.load_state_dict(saved_states["model"])

    if not cfg.gpu.disabled:
        parallel_model = torch.nn.DataParallel(
            model,
            device_ids=cfg.gpu.device_ids,
        )
    else:
        parallel_model = model

    image_shape = model.image_shape
    n_prototypes_per_class = model.prototype_shape[0] // model.n_classes
    maximum_distance = np.prod(model.prototype_shape[1:])

    fold, epoch, prototype_info = get_prototype_information(
        cfg.model_path.stem, dirs, model_train_parameters, logger
    )
    prototype_img_identity = prototype_info[:, -1]
    logger.info(
        f"Prototypes are chosen from {len(set(prototype_img_identity))}"
        f" number of classes."
    )
    logger.info(f"Their class identities are: {prototype_img_identity}")

    prototype_max_connection = torch.argmax(
        model.last_layer.weight, dim=0
    ).cpu().numpy()
    if (np.sum(prototype_max_connection == prototype_img_identity)
            == model.prototype_shape[0]):
        logger.info(
            "All prototypes connect most strongly to their respective classes."
        )
    else:
        logger.warning(
            "Not all prototypes connect most strongly to their respective classes"
        )

    resize = A.Resize(height=image_shape[0], width=image_shape[1])
    preprocess = A.Compose([
        A.ToFloat(max_value=model_train_parameters.data.set.image_properties.max_value),
        A.Normalize(mean=model_train_parameters.data.set.image_properties.mean,
                    std=model_train_parameters.data.set.image_properties.std),
        resize,
        ToTensorV2(),
    ])

    meta_file = initialize_meta_file()

    for image in images:
        test_image_path = image.path
        test_image_name = test_image_path.stem
        test_image_label = image.true_class
        test_image = read_image(test_image_path)
        logger.info(test_image_path)
        meta_file.at[test_image_name, "image_path"] = test_image_path
        meta_file.at[test_image_name, "expected"] = test_image_label

        image_tensor = preprocess(image=test_image)["image"]
        image_variable = Variable(image_tensor.unsqueeze(0))

        test_image = resize(image=test_image)["image"]
        test_image = test_image[..., np.newaxis]

        input_ = image_variable
        if not cfg.gpu.disabled:
            input_ = input_.cuda()

        logits, additional_information = parallel_model(input_)

        prototype_activations = model.distance_2_similarity(
            additional_information.min_distances
        )
        prototype_activation_patterns = model.distance_2_similarity(
            additional_information.distances
        )
        if model.prototype_activation_function == "linear":
            prototype_activations += maximum_distance
            prototype_activation_patterns += maximum_distance

        predicted = torch.argmax(logits, dim=1)[0].item()
        meta_file.at[test_image_name, "predicted"] = test_image_label
        if predicted == test_image_label:
            image_output = dirs.correct
        else:
            image_output = dirs.missed

        for i in range(model.n_classes):
            max_activation_pattern = get_maximum_activation_pattern_of_class_prototypes(
                i, image_shape, logger, model,
                n_prototypes_per_class, predicted,
                prototype_activation_patterns,
                prototype_activations,
                prototype_img_identity,
                prototype_max_connection,
                meta_file,
            )

            heatmap = get_heatmap(max_activation_pattern)
            with_heatmap = np.uint8(test_image * 0.6 + heatmap * 0.4)

            if cfg.outputs.to_save.heatmaps:
                logger.save_image(
                    f"{cfg.outputs.file_prefixes.heatmap}-{test_image_name}"
                    f"-by-class-{i}-prototypes.png",
                    with_heatmap,
                    image_output,
                )
            if cfg.outputs.to_save.max_activation_patch:
                save_highest_activation_patch(
                    image_output,
                    f"{cfg.outputs.file_prefixes.max_activation_patch}-"
                    f"{test_image_name}-by-class-{i}-prototypes.png",
                    logger,
                    max_activation_pattern,
                    test_image
                )
            if cfg.outputs.to_save.max_activation_patch_heatmap:
                save_highest_activation_patch(
                    image_output,
                    f"{cfg.outputs.file_prefixes.max_activation_patch_heatmap}-"
                    f"{test_image_name}-by-class-{i}-prototypes.png",
                    logger,
                    max_activation_pattern,
                    with_heatmap,
                )
            if cfg.outputs.to_save.max_activation_bounding_box:
                save_highest_activation_bounding_box(
                    image_output,
                    f"{cfg.outputs.file_prefixes.max_activation_bounding_box}-"
                    f"{test_image_name}-by-class-{i}-prototypes.png",
                    logger,
                    max_activation_pattern,
                    test_image,
                )

        with logger.increase_indent_context():
            logger.info(f"Predicted: {predicted}")
            logger.info(f"Expected:  {test_image_label}")
            logger.info(f"Saving to: {image_output}")

    meta_file.to_csv(dirs.model / "details.csv")


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIG_PATH"),
    config_name="script_protopnet_visualize_activations",
)
def main(cfg: ScriptConfig):
    logger = ScriptLogger(__name__)

    try:
        cfg = OmegaConf.to_object(cfg)
        run(cfg, logger)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    resolve_create()
    resolve_run_location()
    ConfigStore.instance().store("_script_config_validation", ScriptConfig)

    main()
