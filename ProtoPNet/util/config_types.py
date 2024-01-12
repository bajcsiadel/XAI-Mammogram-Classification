import dataclasses as dc
import numpy as np
import pipe
import typing as typ

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from icecream import ic
from pathlib import Path
from dotenv import load_dotenv

from ProtoPNet.config.backbone_features import BACKBONE_MODELS
from ProtoPNet.util import helpers

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
                raise ValueError(f"Augmentations must be a list of dictionaries. {key} = {value}")
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
                    raise ValueError(f"Number of color channels must be 1 or 3. {key} = {value}")
            case "max_value":
                if value < 1.0:
                    raise ValueError(f"Maximum pixel value must be at least 1.0. {key} = {value}")
            case "mean" | "std":
                if len(value) != self.color_channels:
                    raise ValueError(
                        f"{key[0].upper()}{key[1:]} must have the same number of elements "
                        f"as the number of color channels.\n"
                        f"{self.color_channels = }\n"
                        f"{len(self.mean) = }\n"
                        f"{len(self.std) = }\n"
                    )
                if not np.all(np.array(value) > 0.0):
                    raise ValueError(f"{key[0].upper()}{key[1:]} must be positive.\n{key} = {value}")

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
class Dataset:
    name: str
    root: Path
    size: str
    state: str
    image_dir: Path
    image_properties: ImageProperties
    metadata: MetadataInformation
    number_of_classes: int = 0
    input_size: list[int] = dc.field(default_factory=list)

    def __setattr__(self, key, value):
        match key:
            case "root":
                if not value.exists() or not value.is_dir():
                    raise NotADirectoryError(f"Dataset root {value} does not exist.")
            case "image_dir":
                if not value.exists() or not value.is_dir():
                    raise NotADirectoryError(f"Image directory {value} does not exist.")
            case "size":
                size_values = ["full", "cropped"]
                if value not in size_values:
                    raise ValueError(f"Dataset size {value} not supported. Choose one of {', '.join(size_values)}.")
            case "subset":
                subset_values = ["original", "masked", "preprocessed", "masked_preprocessed"]
                if value not in subset_values:
                    raise ValueError(f"Dataset subset {value} not supported. "
                                     f"Choose one of f{', '.join(subset_values)}.")
            case "number_of_classes" | "input_size":
                if key in self.__dict__:
                    raise ValueError(f"{key} is automatically defined. Should not be set in the configuration file!")

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
    data_filters: list[Filter]
    cross_validation_folds: int
    stratified: bool
    balanced: bool
    grouped: bool
    num_workers: int
    seed: int


@dc.dataclass
class Data:
    set: Dataset
    filters: list[Filter]
    datamodule: DataModule


@dc.dataclass
class Network:
    name: str
    pretrained: bool
    backbone_only: bool
    add_on_layer_type: str

    def __setattr__(self, key, value):
        match key:
            case "name":
                if value not in BACKBONE_MODELS:
                    raise ValueError(f"Network {value} not supported. Choose one of {', '.join(BACKBONE_MODELS)}.")
            case "add_on_layer_type":
                add_on_layer_type_values = ["regular", "bottleneck"]
                if value not in add_on_layer_type_values:
                    raise ValueError(f"Add-on layer type {value} not supported. "
                                     f"Choose one of {', '.join(add_on_layer_type_values)}.")

        super().__setattr__(key, value)


@dc.dataclass
class CrossValidationParameters:
    folds: int
    stratified: bool
    balanced: bool
    grouped: bool

    def __setattr__(self, key, value):
        match key:
            case "folds":
                if value <= 1:
                    raise ValueError(f"Number of cross validation folds must greater than 1. {key} = {value}")
            case "stratified":
                if self.__dict__.get("balanced", False) == value:
                    raise AttributeError(f"Cross validation cannot be both stratified and balanced.")
            case "balanced":
                if self.__dict__.get("stratified", False) == value:
                    raise AttributeError(f"Cross validation cannot be both stratified and balanced.")

        super().__setattr__(key, value)


@dc.dataclass
class PrototypeProperties:
    per_class: int
    size: int
    activation_fn: str

    def __setattr__(self, key, value):
        match key:
            case "per_class":
                if value <= 0:
                    raise ValueError(f"Number of prototypes per class must be positive. {key} = {value}")
            case "size":
                if value <= 0:
                    raise ValueError(f"Prototype size must be positive. {key} = {value}")
            case "activation_fn":
                activation_fn_values = ["log", "linear", "relu", "sigmoid", "tanh"]
                if value not in activation_fn_values:
                    raise ValueError(f"Prototype activation function {value} not supported. "
                                     f"Choose one of {', '.join(activation_fn_values)}.")

        super().__setattr__(key, value)


@dc.dataclass
class Phase:
    batch_size: int = 1
    epochs: int = 0
    learning_rates: dict[str, float] = dc.field(default_factory=dict)
    weight_decay: float = 0.0
    start: int = 0
    interval: int = 0
    scheduler: dict[str, typ.Any] = dc.field(default_factory=dict)
    push_epochs: list[int] = dc.field(default_factory=list)

    def __setattr__(self, key, value):
        match key:
            case "batch_size" | "epochs" | "start" | "interval" | "weight_decay":
                if value < 0:
                    raise ValueError(f"{key[0].upper()}{key[1:]} size must be positive.\n{key} = {value}")
            case "learning_rates":
                for lr_key, lr_value in value.items():
                    if lr_value < 0.0:
                        raise ValueError(f"Learning rate must be positive.\n{lr_key} = {lr_value}")

        super().__setattr__(key, value)


@dc.dataclass
class Phases:
    warm: Phase = dc.field(default_factory=Phase)
    joint: Phase = dc.field(default_factory=Phase)
    push: Phase = dc.field(default_factory=Phase)
    finetune: Phase = dc.field(default_factory=Phase)

    def __post_init__(self):
        self.push.push_epochs = list(range(
            self.push.start,
            self.joint.epochs,
            self.push.interval,
        )) + [self.joint.epochs + 1]


@dc.dataclass
class Loss:
    binary_cross_entropy: bool
    separation_type: str
    coefficients: dict[str, float]

    def __setattr__(self, key, value):
        match key:
            case "separation_type":
                separation_type_values = ["avg", "max", "margin"]
                if value not in separation_type_values:
                    raise ValueError(f"Separation type {self.separation_type} not supported."
                                     f"Choose one of {', '.join(separation_type_values)}.")

        super().__setattr__(key, value)


@dc.dataclass
class JobProperties:
    number_of_workers: int

    def __setattr__(self, key, value):
        match key:
            case "number_of_workers":
                if value <= 0:
                    raise ValueError(f"Number of workers must be greater than 0.\n{key} = {value}")

        super().__setattr__(key, value)


@dc.dataclass
class Gpu:
    ids: list[str]
    disabled: bool = False

    def __setattr__(self, key, value):
        match key:
            case "ids":
                if self.__dict__.get("disabled", False) and len(value) > 0:
                    raise ValueError(f"If GPUs are disabled ids should not be set!")
                gpu_id_values = (list(range(torch.cuda.device_count())) |
                                 pipe.map(str) |
                                 helpers.CustomPipe.to_list)
                for id_ in value:
                    if id_ not in gpu_id_values:
                        raise ValueError(f"GPU id should be between 0 and "
                                         f"{gpu_id_values[-1]}, but {id_} given")
            case "disables":
                if value and len(self.__dict__.get("ids", [])) > 0:
                    raise ValueError(f"If GPUs are disabled ids should not be set!")
            
        super().__setattr__(key, value)


@dc.dataclass
class Dirs:
    model: str
    image: str


@dc.dataclass
class FilePrefixes:
    prototype: str
    self_activation: str
    bounding_box: str


@dc.dataclass
class Outputs:
    dirs: Dirs
    file_prefixes: FilePrefixes


@dc.dataclass
class Config:
    data: Data
    target: str
    network: Network
    prototypes: PrototypeProperties
    cross_validation: CrossValidationParameters
    gpu: Gpu
    seed: int
    loss: Loss
    job: JobProperties
    outputs: Outputs
    phases: Phases

    def __setattr__(self, key, value):
        super().__setattr__(key, value)


def init_config_store():
    config_store_ = ConfigStore.instance()

    config_store_.store(name="_config_validation", node=Config)
    config_store_.store(name="_data_validation", group="data", node=Data)
    config_store_.store(name="_data_set_validation", group="data/set", node=Dataset)
    config_store_.store(name="_cross_validation_validation", group="cross_validation", node=CrossValidationParameters)
    config_store_.store(name="_network_validation", group="network", node=Network)

    return config_store_


@hydra.main(version_base=None, config_path="../conf", config_name="main_config")
def process_config(cfg: Config):
    """
    Process the information in the config file using hydra

    :param cfg: config information read from file
    :return: the processed and validated config
    """
    # print(OmegaConf.to_yaml(cfg))
    cfg: Config = OmegaConf.to_object(cfg)

    ic(cfg.data.set.image_properties.augmentations)
    ic(instantiate(cfg.data.set.image_properties.augmentations.train))
    ic(type(cfg))
    ic(instantiate(cfg.data.datamodule))
    return cfg


if __name__ == "__main__":
    load_dotenv()

    init_config_store()

    process_config()
