import dataclasses as dc

from xai_mam.utils.config.types import Loss, Network


@dc.dataclass
class ProtoPNetLoss(Loss):
    binary_cross_entropy: bool


@dc.dataclass
class PrototypeProperties:
    per_class: int
    size: int
    activation_fn: str
    shape: tuple[int, int, int, int] = dc.field(default_factory=tuple)

    # possible values
    __activation_fn_values = ["log", "linear", "relu", "sigmoid", "tanh"]

    def __setattr__(self, key, value):
        match key:
            case "per_class":
                if value <= 0:
                    raise ValueError(
                        f"Number of prototypes per class must be positive. {key} = {value}"
                    )
            case "size":
                if value <= 0:
                    raise ValueError(
                        f"Prototype size must be positive. {key} = {value}"
                    )
            case "activation_fn":
                if value not in self.__activation_fn_values:
                    raise ValueError(
                        f"Prototype activation function {value} not supported. "
                        f"Choose one of {', '.join(self.__activation_fn_values)}."
                    )

        super().__setattr__(key, value)

    def define_shape(self, n_classes):
        self.shape = (self.per_class * n_classes, self.size, 1, 1)
        return self.shape


@dc.dataclass
class AddOnLayerProperties:
    type: str = "bottleneck"
    activation: str = "B"

    # possible values
    __type_values = ["regular", "bottleneck"]
    __activation_values = ["A", "B"]

    def __setattr__(self, key, value):
        match key:
            case "type":
                if value not in self.__type_values:
                    raise ValueError(
                        f"Add-on layer type {value} not supported. "
                        f"Choose one of {', '.join(self.__type_values)}."
                    )
            case "activation":
                if value not in self.__activation_values:
                    raise ValueError(
                        f"Add-on layer activation {value} not supported. "
                        f"Choose one of {', '.join(self.__activation_values)}."
                    )
        super().__setattr__(key, value)


@dc.dataclass
class ProtoPNetBackboneNetwork(Network):
    add_on_layer_properties: AddOnLayerProperties
