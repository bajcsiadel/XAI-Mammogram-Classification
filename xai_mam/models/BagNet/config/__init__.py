import dataclasses as dc

from xai_mam.models.BagNet import all_models
from xai_mam.utils.config._general_types import ModelParameters, Network
from xai_mam.utils.config.types import Loss


@dc.dataclass
class BagNetLoss(Loss):
    binary_cross_entropy: bool


@dc.dataclass
class BagNetNetwork(Network):
    _network_values = list(all_models.keys()) + ["resnet50"]

    def __setattr__(self, key, value):
        match key:
            case "name":
                if value not in self._network_values:
                    raise ValueError(
                        f"Model {value!r} not supported for explainable "
                        f"BagNet. Choose one of "
                        f"{', '.join(self._network_values)}."
                    )

        super().__setattr__(key, value)


@dc.dataclass
class BagNetParameters(ModelParameters):
    loss: BagNetLoss
