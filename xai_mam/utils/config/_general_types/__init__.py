import dataclasses as dc

from xai_mam.utils.config._general_types._multifunctional import *
from xai_mam.utils.config._general_types.data import *
from xai_mam.utils.config._general_types.log import *
from xai_mam.utils.config._general_types.network import *


@dc.dataclass
class Loss:
    coefficients: dict[str, float]


@dc.dataclass
class ModelParameters:
    construct_trainer: dict
