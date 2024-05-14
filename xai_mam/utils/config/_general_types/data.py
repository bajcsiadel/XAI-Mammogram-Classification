import dataclasses as dc
import typing as typ
from pathlib import Path

import numpy as np

from xai_mam.utils.config._general_types._multifunctional import BatchSize

__all__ = [
    "Augmentation",
    "Augmentations",
    "ImageProperties",
    "CSVParameters",
    "MetadataInformation",
    "Target",
    "Dataset",
    "Filter",
    "Data",
    "DataModule",
]

Augmentation = dict[str, typ.Any]


@dc.dataclass
class Augmentations:
    train: list[Augmentation]
    push: list[Augmentation]
    test: list[Augmentation]

    def __setattr__(self, key, value):
        if not isinstance(value, list):
            raise ValueError(f"Augmentations must be a list. {key} = {value}")

        for augmentation in value:
            if not isinstance(augmentation, dict):
                raise ValueError(
                    f"Augmentations must be a list of dictionaries. {key} = {value}"
                )
            if "_target_" not in augmentation.keys():
                raise ValueError(f"Augmentations must have a _target_. {key} = {value}")

        super().__setattr__(key, value)


@dc.dataclass
class ImageProperties:
    extension: str
    width: int
    height: int
    color_channels: int
    max_value: float
    mean: list[float]
    std: list[float]
    augmentations: Augmentations

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
class CSVParameters:
    index_col: list[int]
    header: list[int]

    def to_dict(self):
        return self.__dict__


@dc.dataclass
class MetadataInformation:
    file: Path
    parameters: CSVParameters

    def __setattr__(self, key, value):
        match key:
            case "file":
                if not value.exists() or not value.is_file() or value.suffix != ".csv":
                    raise FileNotFoundError(f"Metadata file {value} does not exist.")

        super().__setattr__(key, value)


@dc.dataclass
class Target:
    name: str
    size: str


@dc.dataclass
class Dataset:
    name: str
    root: Path
    state: str
    target: Target
    image_dir: Path
    image_properties: ImageProperties
    metadata: MetadataInformation
    number_of_classes: int = 0
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
            case "number_of_classes" | "input_size":
                if key in self.__dict__:
                    raise ValueError(
                        f"{key} is automatically defined. "
                        f"Should not be set in the configuration file!"
                    )

        super().__setattr__(key, value)


@dc.dataclass
class Filter:
    _target_: str
    field: list[str]
    value: typ.Any


@dc.dataclass
class DataModule:
    _target_: str
    data: Dataset
    classification: str
    data_filters: list[Filter] = dc.field(default_factory=list)
    cross_validation_folds: int = 0
    stratified: bool = False
    balanced: bool = False
    grouped: bool = False
    num_workers: int = 0
    seed: int = 1234
    debug: bool = False
    batch_size: BatchSize = dc.field(default_factory=lambda: BatchSize(32, 16))


@dc.dataclass
class Data:
    set: Dataset
    filters: list[Filter]
    datamodule: DataModule


def init_data_config_store():
    from xai_mam.utils.config import config_store_
    config_store_.store(name="_data_validation", group="data", node=Data)
    config_store_.store(name="_data_set_validation", group="data/set", node=Dataset)
