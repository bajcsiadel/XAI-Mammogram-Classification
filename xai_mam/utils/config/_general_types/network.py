import dataclasses as dc
import typing as typ

import hydra

from xai_mam.utils.config._general_types._multifunctional import BatchSize

__all__ = [
    "Network",
    "CrossValidationParameters",
    "Phase",
    "ModelConfig",
]


@dc.dataclass
class Network:
    name: str
    pretrained: bool


@dc.dataclass
class CrossValidationParameters:
    folds: int
    stratified: bool
    balanced: bool
    grouped: bool

    def __setattr__(self, key, value):
        match key:
            case "folds":
                if value < 0:
                    raise ValueError(
                        f"Number of cross validation folds must "
                        f"greater than 1. {key} = {value}"
                    )

        super().__setattr__(key, value)

    def __post_init__(self):
        if self.stratified and self.balanced:
            raise AttributeError(
                "Cross validation cannot be both " "stratified and balanced."
            )


@dc.dataclass
class Optimizer:
    _target_: str = "torch.optim.Adam"
    _args_: list = dc.field(default_factory=list)

    __target_values = ["torch.optim.Adam", "torch.optim.SGD"]

    def __setattr__(self, key, value):
        match key:
            case "_target_":
                if value not in self.__target_values:
                    raise ValueError(
                        f"Optimizer does not support {value!r}. Must be one "
                        f"of the following: {', '.join(self.__target_values)}"
                    )
        super().__setattr__(key, value)


@dc.dataclass
class Phase:
    batch_size: BatchSize = dc.field(default_factory=BatchSize)
    epochs: int = 0
    learning_rates: dict[str, float] = dc.field(default_factory=dict)
    weight_decay: float = 0.0
    scheduler: dict[str, typ.Any] = dc.field(default_factory=dict)
    optimizer: Optimizer = dc.field(default_factory=Optimizer)

    def __setattr__(self, key, value):
        match key:
            case "epochs" | "weight_decay":
                if value < 0:
                    raise ValueError(f"{key} size must be positive.\n{key} = {value}")
            case "learning_rates":
                for lr_key, lr_value in value.items():
                    if lr_value < 0.0:
                        raise ValueError(
                            f"Learning rate must be positive.\n{lr_key} = {lr_value}"
                        )

        super().__setattr__(key, value)


@dc.dataclass
class ModelConfig:
    name: str
    log_parameters_fn: dict
    network: dict
    phases: dict[str, Phase]
    params: dict
    backbone_only: bool = False
    validate_fn: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        self.params = hydra.utils.instantiate(self.params)
        self.network = hydra.utils.instantiate(self.network)

        if "_target_" in self.validate_fn:
            hydra.utils.instantiate(self.validate_fn)(cfg=self)
