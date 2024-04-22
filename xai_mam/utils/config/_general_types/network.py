import dataclasses as dc
import typing as typ

import hydra

from xai_mam.models.utils.backbone_features import all_features
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

    def __setattr__(self, key, value):
        match key:
            case "name":
                if value not in all_features:
                    raise ValueError(
                        f"Network {value} not supported. Choose "
                        f"one of {', '.join(all_features)}."
                    )

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
class Phase:
    batch_size: BatchSize = dc.field(default_factory=BatchSize)
    epochs: int = 0
    learning_rates: dict[str, float] = dc.field(default_factory=dict)
    weight_decay: float = 0.0
    scheduler: dict[str, typ.Any] = dc.field(default_factory=dict)

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
    log_parameters_fn: str
    network: dict
    phases: dict[str, Phase]
    params: dict
    backbone_only: bool = False

    def __post_init__(self):
        self.params = hydra.utils.instantiate(self.params)
        self.network = hydra.utils.instantiate(self.network)

        if self.backbone_only:
            if "warm" in self.phases and self.phases["warm"].epochs != 0:
                raise ValueError(f"Training backbone does not support warmup")
            if "finetune" in self.phases and self.phases["warm"].epochs != 0:
                raise ValueError(f"Training backbone does not support warmup")
            if "push" in self.phases:
                raise ValueError(f"Training backbone does not support push phase")
