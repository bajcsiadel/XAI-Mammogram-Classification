import dataclasses as dc

from xai_mam.utils.config._general_types import ModelParameters, ModelConfig
from xai_mam.utils.config.types import Loss


@dc.dataclass
class BagNetLoss(Loss):
    binary_cross_entropy: bool


@dc.dataclass
class BagNetModel(ModelConfig):
    def __post_init__(self):
        super().__post_init__()


@dc.dataclass
class BagNetParameters(ModelParameters):
    loss: BagNetLoss
