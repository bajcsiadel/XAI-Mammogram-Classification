import dataclasses as dc

from xai_mam.models.ProtoPNet.config import ProtoPNetLoss, PrototypeProperties
from xai_mam.utils.config.types import ModelParameters


@dc.dataclass
class ProtoPNetBackboneParameters(ModelParameters):
    loss: ProtoPNetLoss
    prototypes: PrototypeProperties
