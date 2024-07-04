import dataclasses as dc
import typing as typ
from pathlib import Path

import albumentations as A
import hydra.utils
import numpy as np
from hydra.core.config_store import ConfigStore

from xai_mam.utils import custom_pipe
from xai_mam.utils.config._general_types._multifunctional import BatchSize
from xai_mam.utils.helpers import RepeatedAugmentation

__all__ = [
    "Augmentations",
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
class Augmentations:
    transforms: list[A.BasicTransform | A.Compose | RepeatedAugmentation] = dc.field(
        default_factory=lambda: [A.NoOp]
    )
    online: bool = False

    def __init__(self, transforms: 'AugmentationsConfig' = None):
        """
        Transformations to apply to the images. Converted from the config.

        :param transforms: transforms set in the configuration
        """
        if transforms is None:
            transforms = AugmentationsConfig()
        self.transforms = hydra.utils.instantiate(transforms.transforms)
        self.online = transforms.online

    @property
    def multiplier(self) -> int:
        """
        Get the multiplier for the transformations (number of images generated from
        a base image). If there are no transformations, return ``1``.

        :return: multiplier for the transformations
        """
        multiplier = 1 if len(self.transforms) == 0 else 0
        for transform in self.transforms:
            match transform:
                case RepeatedAugmentation():
                    multiplier += transform.n_repeat
                case A.Compose():
                    multiplier += 1
                case _:
                    return 1
        return multiplier

    @property
    def offline(self) -> bool:
        """
        Check if the transformations are offline.

        :return: ``True`` if the transformations are offline, ``False`` otherwise
        """
        return not self.online

    def get_transforms(self) -> typ.Generator[A.Compose | A.BasicTransform, None, None]:
        """
        Get the transformations to apply.

        :return: generate the transformations to apply one-by-one
        """
        if len(self.transforms) > 0:
            match self.transforms[0]:
                case RepeatedAugmentation() | A.Compose():
                    for transform in self.transforms:
                        yield transform
                case _:
                    yield A.Compose(transforms=[A.Sequential(self.transforms)])
        else:
            yield A.NoOp()

    def get_repetitions(self) -> np.ndarray:
        """
        Get the number of times each transformation is repeated.

        :return: number of times each transformation is repeated
        """
        repetitions = []
        for transform in self.transforms:
            match transform:
                case RepeatedAugmentation():
                    repetitions.append(transform.n_repeat)
                case _:
                    repetitions.append(1)
        return np.array(repetitions)


@dc.dataclass
class AugmentationsConfig:
    transforms: list[Augmentation] = dc.field(
        default_factory=lambda: [{"_target_": "albumentations.NoOp"}]
    )
    exclude_identity_transform: bool = False
    online: bool = False
    __identity_transform_present: bool = False

    def __setattr__(self, key, value):
        match key:
            case "transforms":
                if not isinstance(value, list):
                    raise ValueError(f"Augmentations must be a list. {key} = {value}")
                for augmentation in value:
                    if not isinstance(augmentation, dict) and not issubclass(
                        type(augmentation), (A.BaseCompose, A.BasicTransform)
                    ):
                        raise ValueError(
                            f"Augmentations must be a list of dictionaries. "
                            f"{key} = {value}"
                        )
                    if type(value) is dict and "_target_" not in augmentation.keys():
                        raise ValueError(
                            f"Augmentations must have a _target_. {key} = {value}"
                        )
                self.__identity_transform_present = self.set_identity_transform_present(
                    value
                )

        super().__setattr__(key, value)

    def _validate_augmentations(self):
        compose_augmentations = (
            self.transforms
            | custom_pipe.filter(
                lambda augmentation: augmentation.get("_target_")
                in [
                    "albumentations.Compose",
                    "xai_mam.utils.helpers.RepeatedAugmentation",
                ]
            )
            | custom_pipe.to_list
        )
        if len(compose_augmentations) > 0:
            if len(compose_augmentations) != len(self.transforms):
                raise ValueError(
                    "Mixing RepeatedAugmentation/Compose "
                    "and BasicTransforms is not allowed."
                )
            elif (
                not self.__identity_transform_present
                and not self.exclude_identity_transform
            ):
                # if there are multiple transforms then add a transform
                # to keep the original image
                self.transforms.append(
                    {
                        "_target_": "albumentations.Compose",
                        "transforms": [{"_target_": "albumentations.NoOp"}],
                    }
                )
                self.__identity_transform_present = True
        else:
            self.exclude_identity_transform = True
            self.transforms = [
                {
                    "_target_": "albumentations.Compose",
                    "transforms": self.transforms,
                }
            ]

    def set_identity_transform_present(self, transforms):
        for augmentation in transforms:
            if augmentation.get("_target_") == "albumentations.NoOp":
                return True
            if (child_transforms := augmentation.get("transforms")) is not None:
                if self.set_identity_transform_present(child_transforms):
                    return True
        return False

    def __post_init__(self):
        self.__identity_transform_present = self.set_identity_transform_present(
            self.transforms
        )
        self._validate_augmentations()

    def to_instance(self) -> Augmentations:
        """
        Converts the current augmentation configuration to an instance.

        :return: instance of the augmentations
        """
        return Augmentations(self)


@dc.dataclass
class AugmentationGroupsConfig:
    train: AugmentationsConfig = dc.field(default_factory=AugmentationsConfig)
    validation: AugmentationsConfig = dc.field(default_factory=AugmentationsConfig)


@dc.dataclass
class ImagePropertiesConfig:
    extension: str
    width: int
    height: int
    n_color_channels: int
    max_value: float
    mean: list[float]
    std: list[float]
    augmentations: AugmentationGroupsConfig = dc.field(
        default_factory=AugmentationGroupsConfig
    )

    def __setattr__(self, key, value):
        match key:
            case "extension":
                if value[0] != ".":
                    value = "." + value
            case "width" | "height":
                if value <= 0:
                    raise ValueError(f"Image {key} must be positive. {key} = {value}")
            case "n_color_channels":
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
                if len(value) != self.n_color_channels:
                    raise ValueError(
                        f"{key} must have the same number of elements "
                        f"as the number of color channels.\n"
                        f"{self.n_color_channels = }\n"
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

    @staticmethod
    def init_store(
        config_store_: ConfigStore = None, group: str = "data/set"
    ) -> ConfigStore:
        if config_store_ is None:
            from xai_mam.utils.config import config_store_

        config_store_.store(
            name="_data_set_validation", group=group, node=DatasetConfig
        )

        return config_store_


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
    n_workers: int = 0
    seed: int = 1234
    debug: bool = False
    batch_size: BatchSize = dc.field(default_factory=lambda: BatchSize(32, 16))
    _convert_: str = "object"  # Structured Configs are converted to instances
    _recursive_: bool = False


@dc.dataclass
class DataConfig:
    set: DatasetConfig
    datamodule: DataModuleConfig
    filters: list[FilterConfig] = dc.field(default_factory=list)

    @staticmethod
    def init_store(
        config_store_: ConfigStore = None, group: str = "data"
    ) -> ConfigStore:
        if config_store_ is None:
            from xai_mam.utils.config import config_store_

        config_store_.store(name="_data_validation", group=group, node=DataConfig)
        DatasetConfig.init_store(config_store_)

        return config_store_
