import albumentations as A
import dataclasses as dc
import typing as typ
from pathlib import Path

import numpy as np

from xai_mam.utils import custom_pipe
from xai_mam.utils.config._general_types._multifunctional import BatchSize

__all__ = [
    "Augmentation",
    "AugmentationsConfig",
    "AugmentationGroupsConfig",
    "ImagePropertiesConfig",
    "CSVParametersConfig",
    "MetadataInformationConfig",
    "TargetConfig",
    "DatasetConfig",
    "FilterConfig",
    "DataConfig",
    "DataModuleConfig",
]

Augmentation = dict[str, typ.Any]


@dc.dataclass
class AugmentationsConfig:
    transforms: list[Augmentation] = dc.field(
        default_factory=lambda: [{"_target_": "albumentations.NoOp"}]
    )
    __identity_transform_present: bool = False

    def __setattr__(self, key, value):
        match key:
            case "transforms":
                if not isinstance(value, list):
                    raise ValueError(f"Augmentations must be a list. {key} = {value}")
                for augmentation in value:
                    if (not isinstance(augmentation, dict) and
                            not issubclass(type(augmentation),
                                           (A.BaseCompose, A.BasicTransform))):
                        raise ValueError(
                            f"Augmentations must be a list of dictionaries. {key} = {value}"
                        )
                    if type(value) is dict and "_target_" not in augmentation.keys():
                        raise ValueError(
                            f"Augmentations must have a _target_. {key} = {value}")
                    self.__identity_transform_present = self.set_identity_transform_present(value)

        super().__setattr__(key, value)

    def _validate_augmentations(self):
        compose_augmentations = (self.transforms
                                 | custom_pipe.filter(
                    lambda augmentation: augmentation.get(
                        "_target_") in ["albumentations.Compose",
                                        "xai_mam.utils.helpers.RepeatedAugmentation"])
                                 | custom_pipe.to_list)
        if len(compose_augmentations) > 0:
            if len(compose_augmentations) != len(self.transforms):
                raise ValueError("Mixing RepeatedAugmentation/Compose "
                                 "and BasicTransforms is not allowed.")
            elif not self.__identity_transform_present:
                # if there are multiple transforms then add a transform
                # to keep the original image
                self.transforms.append({
                    "_target_": "albumentations.Compose",
                    "transforms": [{
                        "_target_": "albumentations.NoOp"
                    }]
                })
        else:
            # TODO: add a flag here to skip identity transform if needed
            # convert BasicTransforms to Compose
            self.transforms = [{
                "_target_": "albumentations.Compose",
                "transforms": self.transforms,
            }]

    def set_identity_transform_present(self, transforms):
        for augmentation in transforms:
            if augmentation.get("_target_") == "albumentations.NoOp":
                return True
            if (child_transforms := augmentation.get("transforms")) is not None:
                return self.set_identity_transform_present(child_transforms)
        return False

    def __post_init__(self):
        self.__identity_transform_present = self.set_identity_transform_present(self.transforms)
        self._validate_augmentations()


@dc.dataclass
class AugmentationGroupsConfig:
    train: AugmentationsConfig = dc.field(default_factory=AugmentationsConfig)
    push: AugmentationsConfig = dc.field(default_factory=AugmentationsConfig)


@dc.dataclass
class ImagePropertiesConfig:
    extension: str
    width: int
    height: int
    color_channels: int
    max_value: float
    mean: list[float]
    std: list[float]
    augmentations: AugmentationGroupsConfig = dc.field(default_factory=AugmentationGroupsConfig)

    def __setattr__(self, key, value):
        match key:
            case "extension":
                if value[0] != ".":
                    value = "." + value
            case "width" | "height":
                if value <= 0:
                    raise ValueError(f"Image {key} must be positive. {key} = {value}")
            case "color_channels":
                if value not in [1, 3]:
                    raise ValueError(
                        f"Number of color channels must be 1 or 3. {key} = {value}"
                    )
            case "max_value":
                if value < 1.0:
                    raise ValueError(
                        f"Maximum pixel value must be at least 1.0. {key} = {value}"
                    )
            case "mean" | "std":
                if len(value) != self.color_channels:
                    raise ValueError(
                        f"{key} must have the same number of elements "
                        f"as the number of color channels.\n"
                        f"{self.color_channels = }\n"
                        f"{len(self.mean) = }\n"
                        f"{len(self.std) = }\n"
                    )
                if not np.all(np.array(value) > 0.0):
                    raise ValueError(f"{key} must be positive.\n{key} = {value}")

        super().__setattr__(key, value)


@dc.dataclass
class CSVParametersConfig:
    index_col: list[int]
    header: list[int]

    def to_dict(self):
        return self.__dict__


@dc.dataclass
class MetadataInformationConfig:
    file: Path
    parameters: CSVParametersConfig

    def __setattr__(self, key, value):
        match key:
            case "file":
                if not value.exists() or not value.is_file() or value.suffix != ".csv":
                    raise FileNotFoundError(f"Metadata file {value} does not exist.")

        super().__setattr__(key, value)


@dc.dataclass
class TargetConfig:
    name: str
    size: str


@dc.dataclass
class DatasetConfig:
    name: str
    root: Path
    state: str
    target: TargetConfig
    image_dir: Path
    image_properties: ImagePropertiesConfig
    metadata: MetadataInformationConfig
    number_of_classes: int = 0  # set automatically from code
    input_size: tuple[int, int] = (0, 0)

    # possible values
    __subset_values = ["original", "masked", "preprocessed", "masked_preprocessed"]
    __size_values = ["full", "cropped"]

    def __setattr__(self, key, value):
        match key:
            case "root":
                if not value.exists() or not value.is_dir():
                    raise NotADirectoryError(f"Dataset root {value} does not exist.")
            case "image_dir":
                if not value.exists() or not value.is_dir():
                    raise NotADirectoryError(f"Image directory {value} does not exist.")
            case "size":
                if value not in self.__size_values:
                    raise ValueError(
                        f"Dataset size {value} not supported. "
                        f"Choose one of {', '.join(self.__size_values)}."
                    )
            case "subset":
                if value not in self.__subset_values:
                    raise ValueError(
                        f"Dataset subset {value} not supported. "
                        f"Choose one of f{', '.join(self.__subset_values)}."
                    )
            case "number_of_classes":
                limit = -1
                if key in self.__dict__:
                    limit = 0
                if value <= limit:
                    raise ValueError(f"{key} must be positive. {key} = {value}")

        super().__setattr__(key, value)


@dc.dataclass
class FilterConfig:
    _target_: str
    field: list[str]
    value: typ.Any


@dc.dataclass
class DataModuleConfig:
    _target_: str
    data: DatasetConfig
    classification: str
    data_filters: list[FilterConfig] = dc.field(default_factory=list)
    cross_validation_folds: int = 0
    stratified: bool = False
    balanced: bool = False
    grouped: bool = False
    num_workers: int = 0
    seed: int = 1234
    debug: bool = False
    batch_size: BatchSize = dc.field(default_factory=lambda: BatchSize(32, 16))
    _convert_: str = "object"  # Structured Configs are converted to instances
    _recursive_: bool = False


@dc.dataclass
class DataConfig:
    set: DatasetConfig
    filters: list[FilterConfig]
    datamodule: DataModuleConfig


def init_data_config_store():
    from xai_mam.utils.config import config_store_
    config_store_.store(name="_data_validation", group="data", node=DataConfig)
    config_store_.store(name="_data_set_validation", group="data/set", node=DatasetConfig)
