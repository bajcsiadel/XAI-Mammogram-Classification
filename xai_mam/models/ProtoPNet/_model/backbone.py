import torch
from torch import nn

from xai_mam.models._base_classes import Backbone
from xai_mam.models.ProtoPNet._model import ProtoPNetBase


class ProtoPNetBackbone(ProtoPNetBase, Backbone):
    """
    Corresponding backbone feature of an explainable ProtoPNet model.

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
    :param n_color_channels: number of color channels in the input. Defaults to ``3``.
    :type n_color_channels: int
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
        n_color_channels=3,
        add_on_layers_type="bottleneck",
        add_on_layers_activation="A",
    ):
        super(ProtoPNetBackbone, self).__init__(
            features,
            img_shape,
            prototype_shape,
            n_classes,
            logger,
            add_on_layers_type,
            add_on_layers_activation,
        )

        x = torch.randn(8, n_color_channels, *img_shape)

        x = self.features(x)
        x = self.add_on_layers(x)
        _, d, w, h = x.size()
        self.last_layer = nn.Linear(
            d * w * h, self._n_classes, bias=False
        )  # do not use bias

        self.logger.create_csv_log(
            "train_model",
            ("fold", "epoch", "phase"),
            "time",
            "cross entropy",
            "accuracy",
            "precision",
            "recall",
            "micro_f1",
            "macro_f1",
            "l1",
            exist_ok=True,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        :param x:
        :type x: torch.Tensor
        :return: result of the network on the given data
        :rtype: torch.Tensor
        """
        x = self.features(x)
        x = self.add_on_layers(x)
        x = x.view(x.size()[0], -1)
        return self.last_layer(x)

    def __repr__(self):
        """
        String representation of the network.

        :return: string representation of the network
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"\timg_shape: {self._image_shape},\n"
            f"\tnum_classes: {self._n_classes},\n"
            f"\tepsilon: {self._epsilon}\n)\n"
            f"{super(ProtoPNetBackbone, self).__repr__()}"
        )
