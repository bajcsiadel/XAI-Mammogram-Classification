from abc import ABC, abstractmethod

import torch.nn as nn

from xai_mam.utils.errors import AbstractClassError


class _AddOnLayers(nn.Sequential, ABC):
    """
    Base class for the add-on layers.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    :param activation_type: type of activation between the convolutional layers
    :type activation_type: str

    :raises AbstractClassError: if class with ABC super class is
        instantiated directly
    """

    def __new__(cls, *args, **kwargs):
        if ABC.__name__ in [super_.__name__ for super_ in cls.__bases__]:
            raise AbstractClassError(
                f"Can not create an instance " f"of {cls.__name__} directly"
            )
        return super(_AddOnLayers, cls).__new__(cls)

    def __init__(self, in_channels, out_channels, activation_type="A"):
        super(_AddOnLayers, self).__init__()

        self._activation = A() if activation_type == "A" else B()

        self._in_channels = in_channels
        self._out_channels = out_channels


class Bottleneck(_AddOnLayers, ABC):
    """
    Class representing the bottleneck add-on layer.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    :param activation_type: type of activation between the convolutional layers
    :type activation_type: str
    """

    def __init__(self, in_channels, out_channels, activation_type="A"):
        super(Bottleneck, self).__init__(
            in_channels,
            out_channels,
            activation_type,
        )

        current_in_channels = in_channels
        first = True
        i = 1
        while current_in_channels > out_channels or first:
            current_out_channels = max(out_channels, current_in_channels // 2)
            self.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_channels=current_in_channels,
                    out_channels=current_out_channels,
                    kernel_size=1,
                ),
            )

            self._activation.add_activation(self, i, locals())

            i += 1

            self.add_module(
                f"conv{i}",
                nn.Conv2d(
                    in_channels=current_out_channels,
                    out_channels=current_out_channels,
                    kernel_size=1,
                ),
            )
            if current_out_channels > out_channels:
                self._activation.add_activation(self, i, locals())
            else:
                assert current_out_channels == out_channels
                self.add_module("sigmoid", nn.Sigmoid())
            current_in_channels = current_in_channels // 2
            first = False


class Regular(_AddOnLayers, ABC):
    """
    Class representing the regular add-on layer.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    :param activation_type: type of activation between the convolutional layers
    :type activation_type: str
    """

    def __init__(self, in_channels, out_channels, activation_type="A"):
        super(Regular, self).__init__(
            in_channels,
            out_channels,
            activation_type,
        )

        self.add_module(
            "conv1",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

        self._activation.add_activation(self, 1, locals())

        self.add_module(
            "conv2",
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )
        self.add_module("sigmoid", nn.Sigmoid())


class Pool(_AddOnLayers, ABC):
    """
    A simple add-on layer consisting of adaptive average pooling.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    :param activation_type: type of activation between the convolutional layers
    :type activation_type: str
    """

    def __init__(self, in_channels, out_channels, activation_type="A"):
        super(Pool, self).__init__(
            in_channels,
            out_channels,
            activation_type,
        )

        self.add_module(
            "avgpool",
            nn.AdaptiveAvgPool2d((1, 1)),
        )


class _ActivationType:
    """
    Base class for the activation types.
    """

    _get_params = {}

    def __get_current_params(self, locals_vars):
        """
        Get the current parameters for the activation function.

        :param locals_vars: local variables from the parent function
        :type locals_vars: dict[str, Any]
        :return: parameters defined based on the current variable values
        :rtype: dict[str, Any]
        """
        result = {}
        for key, values in self._get_params.items():
            if isinstance(values, str):
                result[key] = locals_vars.get(values)
            else:
                i = 0
                # search the value in the locals stop at the first value found
                while (value := locals_vars.get(values[i])) is None and i < len(values):
                    i += 1
                result[key] = value
        return result

    def add_activation(self, instance, number, local_vars):
        """
        Call the function to add the activation function between the
        convolutions to the instance.

        :param instance: add-on module instance
        :type instance: nn.Module
        :param number: the number of the activation
        :type number: int
        :param local_vars: local variables from the parent function
        :type local_vars: dict[str, Any]
        """
        self._add_activation(instance, number, **self.__get_current_params(local_vars))

    @abstractmethod
    def _add_activation(self, instance, number, **kwargs):
        """
        Add the activation between the convolutions to the instance.

        :param instance: add-on module instance
        :type instance: nn.Module
        :param number: the number of the activation
        :type number: int
        :param kwargs:
        """
        ...

    @classmethod
    @property
    @abstractmethod
    def citation(cls):
        """
        Get the cite of the article in which the activation type is presented.

        :return: the citation for the activation type
        :rtype: str
        """
        ...


class A(_ActivationType):
    """
    Class representing the activation type A presented in the
    original article [Chen+18]_. It consists of a singe ``ReLU`` layer.

    .. [Chen+18] Chaofan Chen et al. "This looks like that: deep learning
        for interpretable image recognition".
        Url: `http://arxiv.org/abs/1806.10574
        <http://arxiv.org/abs/1806.10574>`_
    """

    def _add_activation(self, instance, number, **kwargs):
        """
        Add the activation between the convolutions to the instance.

        :param instance: add-on module instance
        :type instance: nn.Module
        :param number: the number of the activation
        :type number: int
        :param kwargs:
        """
        instance.add_module(f"relu{number}", nn.ReLU())

    @classmethod
    @property
    def citation(cls):
        """
        Get the cite of the article in which the activation type is presented.

        :return: the citation for the activation type
        :rtype: str
        """
        return "[Chen+18]"


class B(_ActivationType):
    """
    Class representing the activation type B presented in article [Car+22]_,
    where ProtoPNet is applied for mammograms. It consists of a 2-dimensional
    batch normalization, ``ReLU`` activation and a 2-dimensional dropout layer.

    .. [Car+22] Gianluca Carloni et al. "On the Applicability of Prototypical
        Part Learning in Medical Images: Breast Masses Classification Using
        ProtoPNet". `doi:10.1007/978-3-031-37660-3_38
        <https://doi.org/10.1007/978-3-031-37660-3_38>`_
    """

    _get_params = {"num_features": ("current_out_channels", "out_channels")}

    def _add_activation(self, instance, number, **kwargs):
        """
        Add the activation between the convolutions to the instance.

        :param instance: add-on module instance
        :type instance: nn.Module
        :param number: the number of the activation
        :type number: int
        :param kwargs:
        """
        instance.add_module(f"bn{number}", nn.BatchNorm2d(kwargs["num_features"]))
        instance.add_module(f"relu{number}", nn.ReLU())
        instance.add_module(f"dropout{number}", nn.Dropout2d())

    @classmethod
    @property
    def citation(cls):
        """
        Get the cite of the article in which the activation type is presented.

        :return: the citation for the activation type
        :rtype: str
        """
        return "[Car+22]"


class AddOnLayers:
    """
    Class collecting functions related to add-on layers.
    """

    @staticmethod
    def with_type(name, in_channels, out_channels, option="A"):
        """
        Create an add-on layer instance with the given name and option.

        :param name: name of the add-on layer type
        :type name: str
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param option: type of the activation between the convolutional layers
        :type option: str
        :return: add-on layer instance
        :rtype: _AddOnLayers
        """
        class_name = f"{name.capitalize()}{option.capitalize()}"
        add_on_layer_class = globals().get(class_name)

        if add_on_layer_class is None:
            raise ValueError(f"Add on layer type with name " f"{class_name} not found")

        return add_on_layer_class(in_channels, out_channels)

    @staticmethod
    def get_reference(option="A"):
        """
        Get the reference where the activation type was presented.

        :param option: type of the activation between the convolutional layers
        :type option: str
        :return: short reference
        :rtype: str
        """
        match option:
            case "A":
                return A.citation
            case "B":
                return B.citation
            case _:
                raise ValueError(f"Activation type with name {option} not found")


class BottleneckA(Bottleneck):
    """
    Class representing the bottleneck add-on layer with activation type A.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    """

    def __init__(self, in_channels, out_channels):
        super(BottleneckA, self).__init__(in_channels, out_channels, "A")


class BottleneckB(Bottleneck):
    """
    Class representing the bottleneck add-on layer with activation type B.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    """

    def __init__(self, in_channels, out_channels):
        super(BottleneckB, self).__init__(in_channels, out_channels, "B")


class RegularA(Regular):
    """
    Class representing the regular add-on layer with activation type A.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    """

    def __init__(self, in_channels, out_channels):
        super(RegularA, self).__init__(in_channels, out_channels, "A")


class RegularB(Regular):
    """
    Class representing the regular add-on layer with activation type B.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    """

    def __init__(self, in_channels, out_channels):
        super(RegularB, self).__init__(in_channels, out_channels, "B")


class PoolA(Pool):
    """
    Class representing the average pool add-on layer with activation type A.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    """

    def __init__(self, in_channels, out_channels):
        super(PoolA, self).__init__(in_channels, out_channels, "A")


class PoolB(Pool):
    """
    Class representing the average pool add-on layer with activation type B.

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    """

    def __init__(self, in_channels, out_channels):
        super(PoolB, self).__init__(in_channels, out_channels, "B")


if __name__ == "__main__":
    a = BottleneckA(256, 32)
    print(a)
    b = BottleneckB(256, 32)
    print(b)
    a = RegularA(256, 32)
    print(a)
    b = RegularB(256, 32)
    print(b)
    a = PoolA(256, 32)
    print(a)
    b = PoolB(256, 32)
    print(b)

    print(AddOnLayers.with_type("bottleneck", 256, 32, "A"))
    print(AddOnLayers.with_type("bottleneck", 256, 32, "B"))
    print(AddOnLayers.with_type("regular", 256, 32, "A"))
    print(AddOnLayers.with_type("regular", 256, 32, "B"))
    print(AddOnLayers.with_type("pool", 256, 32, "A"))
    print(AddOnLayers.with_type("pool", 256, 32, "B"))
