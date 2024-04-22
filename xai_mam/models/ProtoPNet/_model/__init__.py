from abc import ABC, abstractmethod

import torch
from torch import nn

from xai_mam.models._base_classes import Model
from xai_mam.models.ProtoPNet._model._add_on_layers import AddOnLayers


class _PositiveLinear(nn.Module):
    """
    Linear layer with (only) positive weights, by using the exponential of the weights.

    :param in_features: number of input features
    :type in_features: int
    :param out_features: number of output features
    :type out_features: int
    :param bias: Defaults to ``True``.
    :type bias: bool

    :raises NotImplementedError: it is not yet implemented to set ``bias`` to ``False``
    """

    def __init__(self, in_features, out_features, bias=True):
        super(_PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            raise NotImplementedError("Positive bias is not implemented.")
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters of the layer.
        """
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            raise NotImplementedError("Bias initialization is not implemented.")

    def forward(self, x):
        """
        Forward pass of the layer.

        :param x: input of the layer
        :type x: torch.Tensor
        :return:
        :rtype: torch.Tensor
        """
        return nn.functional.linear(x, self.weight.exp(), self.bias)


class ProtoPNetBase(Model, ABC):
    """
    Base class for the xai_mam and ProtoPNetBackbone models.

    :param features: features of the backbone architecture,
        responsible for the feature extraction
    :type features: nn.Module
    :param img_shape: shape of the input image
    :type img_shape: (int, int)
    :param prototype_shape: shape of the prototype tensor
    :type prototype_shape: (int, int, int, int)
    :param n_classes: number of classes in the data
    :type n_classes: int
    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    :param add_on_layers_type: type of the add-on layers.
        Defaults to ``"bottleneck"``.
    :type add_on_layers_type: str
    :param add_on_layers_activation: activation function for the add-on layers.
        Defaults to ``"A"``.
    :type add_on_layers_activation: str
    """

    def __init__(
        self,
        features,
        img_shape,
        prototype_shape,
        n_classes,
        logger,
        add_on_layers_type="bottleneck",
        add_on_layers_activation="A",
    ):
        super(ProtoPNetBase, self).__init__()
        self._image_shape = img_shape
        self._prototype_shape = prototype_shape
        self._n_classes = n_classes
        self._epsilon = 1e-4

        self.logger = logger

        self.features = features
        features_name = str(self.features).upper()
        if features_name.startswith("VGG") or features_name.startswith("RES"):
            first_add_on_layer_in_channels = [
                i for i in self.features.modules() if isinstance(i, nn.Conv2d)
            ][-1].out_channels
        elif features_name.startswith("DENSE"):
            first_add_on_layer_in_channels = [
                i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)
            ][-1].num_features
        else:
            raise Exception("other base base_architecture NOT implemented")

        self.add_on_layers = AddOnLayers.with_type(
            add_on_layers_type,
            first_add_on_layer_in_channels,
            self._prototype_shape[1],
            add_on_layers_activation,
        )

    @property
    def prototype_shape(self):
        """
        Returns the shape of the prototypes.

        :return: shape of the prototypes
        :rtype: tuple[int, int, int, int]
        """
        return self._prototype_shape

    @property
    def n_classes(self):
        """
        Returns the number of classes in the classification.

        :return: number of classes
        :rtype: int
        """
        return self._n_classes

    @property
    def epsilon(self):
        """
        Returns the ``epsilon`` parameter.

        :return: epsilon
        :rtype: float
        """
        return self._epsilon

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the network.

        :param x:
        :type x: torch.Tensor
        :return: result of the network on the given data
        :rtype: torch.Tensor
        """
        ...

    def __repr__(self):
        """
        Get the representation of the model (layers and parameters).

        :return: module structure and parameters
        :rtype: str
        """
        return super().__repr__()
