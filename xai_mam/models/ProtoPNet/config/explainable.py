import dataclasses as dc

from xai_mam.models.ProtoPNet.config import ProtoPNetLoss, PrototypeProperties
from xai_mam.utils.config.types import ModelParameters


@dc.dataclass
class PushSettings:
    batch_size: int = 16
    start: int = 1
    interval: int = 5
    push_epochs: tuple[int, ...] = dc.field(default_factory=tuple)

    def __setattr__(self, key, value):
        match key:
            case "batch_size" | "start" | "interval":
                if value <= 0:
                    raise ValueError(f"{key!r} must be higher than 0")

        super().__setattr__(key, value)

    def define_push_epochs(self, epochs):
        self.push_epochs = tuple(
            list(range(self.start, epochs, self.interval)) + [epochs]
        )
        return self.push_epochs


@dc.dataclass
class ProtoPNetExplainableLoss(ProtoPNetLoss):
    separation_type: str

    # possible values
    __separation_type_values = ["avg", "max", "margin"]

    def __setattr__(self, key, value):
        match key:
            case "separation_type":
                if value not in self.__separation_type_values:
                    raise ValueError(
                        f"Separation type {self.separation_type} not supported."
                        f"Choose one of {', '.join(self.__separation_type_values)}."
                    )

        super().__setattr__(key, value)


@dc.dataclass
class ProtoPNetExplainableParameters(ModelParameters):
    push: PushSettings
    prototypes: PrototypeProperties
    loss: ProtoPNetExplainableLoss
    class_specific: bool
