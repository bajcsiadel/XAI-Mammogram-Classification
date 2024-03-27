import dataclasses as dc

from ProtoPNet.models.ProtoPNet.config import ProtoPNetLoss, PrototypeProperties
from ProtoPNet.utils.config.types import ModelParameters


@dc.dataclass
class ProtoPNetBackboneParameters(ModelParameters):
    loss: ProtoPNetLoss
    prototypes: PrototypeProperties
