import dataclasses as dc

from ProtoPNet.utils.config._general_types._multifunctional import *
from ProtoPNet.utils.config._general_types.data import *
from ProtoPNet.utils.config._general_types.log import *
from ProtoPNet.utils.config._general_types.network import *


@dc.dataclass
class Loss:
    coefficients: dict[str, float]


@dc.dataclass
class ModelParameters:
    construct_trainer: dict
