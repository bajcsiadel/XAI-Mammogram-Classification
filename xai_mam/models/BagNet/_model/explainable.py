import math

import torch
import torch.nn as nn

from xai_mam.models._base_classes import Explainable
from xai_mam.models.BagNet._model import BagNetBase
from xai_mam.models.utils.backbone_features.resnet_features import (
    BaseResNet,
    Bottleneck,
    ResidualBlock,
)
from xai_mam.models.utils.helpers import get_state_dict
from xai_mam.utils.log import TrainLogger

__base_url = (
    "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models"
    "/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8"
)
__model_urls = {
    "bagnet9": f"{__base_url}/bagnet8-34f4ccd2.pth.tar",
    "bagnet17": f"{__base_url}/bagnet16-105524de.pth.tar",
    "bagnet33": f"{__base_url}/bagnet32-2ddd53ed.pth.tar",
}


class BagNet(BagNetBase, Explainable):
    """
    BagNet model [Bren+19]_ for prototypical image recognition.

    .. [Bren+19] Wieland Brendel and Matthias Bethge, "Approximating CNNs with
        Bag-of-local-Features models works surprisingly well on ImageNet".
        Url: `https://arxiv.org/abs/1904.00760
        <https://arxiv.org/abs/1904.00760>`_

    :param block: type of the residual block used in the model
    :param layers: number of layers in a residual block
    :param n_classes: number of classes in the data
    :param logger:
    :param n_color_channels: number of color channels in the data. Defaults to ``3``.
    :param channels: number of output channels of the stem. Defaults to ``64``.
    :param channels_per_layer: number of output channels of each residual block.
        Defaults to ``None`` ==> ``[64,128,256,512]``.
    :param kernels: kernel sizes for the blocks in each residual block layers.
        Defaults to ``None`` ==> ``[[3,1,1],[3,1,1],[3,1,1],[1,1,1]]`` if
        ``layers`` is ``[3,3,3,3]``.
    :param strides: stride of the first block in each residual block layers.
        Defaults to ``None`` ==> ``[2,2,2,1]``
    :param paddings: padding of the first block in each residual block layers.
        Defaults to ``None`` ==> ``[0,0,0,0]``
    :param avg_pool: marks if global average pooling is used. Defaults to ``False``.
    :param n_kernel_3x3: number of 3x3 kernels in the residual block. Defaults to ``3``.
    """

    def __init__(
        self,
        block: type[ResidualBlock],
        layers: list[int],
        n_classes: int,
        logger: TrainLogger,
        n_color_channels: int = 3,
        channels: int = 64,
        channels_per_layer: list[int] | None = None,
        kernels: list[int] | None = None,
        strides: list[int] | None = None,
        paddings: list[int] | None = None,
        avg_pool: bool = True,
        n_kernel_3x3: int = 3,
    ):
        super(BagNet, self).__init__(n_classes, logger, n_color_channels)
        self.conv1 = nn.Conv2d(
            n_color_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)

        if kernels is None:
            kernels = [[1] * num_blocks for num_blocks in layers]
            if n_kernel_3x3 > len(layers):
                raise ValueError(
                    f"Number of 3x3 kernels ({n_kernel_3x3}) should be "
                    f"less than the number of layers ({len(layers)})."
                )
            for num_blocks in range(n_kernel_3x3):
                kernels[num_blocks][0] = 3

        if strides is None:
            strides = [2, 2, 2, 1]

        if paddings is None:
            paddings = [0] * len(layers)

        self.residual_blocks = BaseResNet(
            block, layers, channels, channels_per_layer, kernels, strides, paddings
        )

        self.fc = nn.Linear(self.residual_blocks.layer4[-1].out_channels, n_classes)
        self.avg_pool = avg_pool
        self.block = block
        self.num_classes = n_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        """
        Push the `x` input forward on the stem of the module.

        :param x:
        :return: the result of the stem
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Push the `x` input forward on the residual block (feature extraction).

        :param x:
        :return: the extracted features
        """
        return self.residual_blocks(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Push the `x` input forward on the module.

        :param x:
        :return: the result of the module
        """
        x = self._forward_stem(x)
        x = self._forward_features(x)

        if self.avg_pool:
            x = nn.AvgPool2d((x.size()[2], x.size()[3]), stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)

        return x


def _load_pretrained_model_wights(
    model: BagNet,
    model_name: str,
    color_channels: int = 3,
    n_classes: int | None = None,
) -> BagNet:
    """
    Load pretrained wights to the given model.

    :param model:
    :param model_name:
    :param color_channels:
    :param n_classes: number of classes to train
    :return:
    """
    pretrained_state_dict = get_state_dict(
        __model_urls[model_name],
        color_channels=color_channels,
        prefixes={r"layer[0-9]\.*": "residual_blocks"},
        include_fc=True,
        n_classes=n_classes,
    )
    model.load_state_dict(pretrained_state_dict)

    return model


def bagnet33(
    n_classes: int,
    logger: TrainLogger,
    n_color_channels: int = 3,
    pretrained: bool = False,
    strides: list[int] | None = None,
    **kwargs,
) -> BagNet:
    """
    Constructs a Bagnet-33 model.

    :param n_classes: number of classes.
    :param logger:
    :param n_color_channels: number of color channels. Defaults to ``3``.
    :param pretrained: If ``True``, returns a model pre-trained on ImageNet.
        Defaults to ``False``.
    :param strides: Strides of the first layer of each residual block.
        Defaults to ``None``.
    :return: A Bagnet-33 model.
    """
    model = BagNet(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes=n_classes,
        logger=logger,
        n_color_channels=n_color_channels,
        strides=strides,
        n_kernel_3x3=4,
        **kwargs,
    )
    if pretrained:
        model = _load_pretrained_model_wights(
            model, "bagnet33", n_color_channels, n_classes
        )
    return model


def bagnet17(
    n_classes: int,
    logger: TrainLogger,
    n_color_channels: int = 3,
    pretrained: bool = False,
    strides: list[int] | None = None,
    **kwargs,
) -> BagNet:
    """
    Constructs a Bagnet-17 model.

    :param n_classes: number of classes.
    :param logger:
    :param n_color_channels: number of color channels. Defaults to ``3``.
    :param pretrained: If ``True``, returns a model pre-trained on ImageNet.
        Defaults to ``False``.
    :param strides: Strides of the first layer of each residual block.
        Defaults to ``None``.
    :return: A Bagnet-17 model.
    """
    model = BagNet(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes=n_classes,
        logger=logger,
        n_color_channels=n_color_channels,
        strides=strides,
        n_kernel_3x3=3,
        **kwargs,
    )
    if pretrained:
        model = _load_pretrained_model_wights(
            model, "bagnet17", n_color_channels, n_classes
        )
    return model


def bagnet9(
    n_classes: int,
    logger: TrainLogger,
    n_color_channels: int = 3,
    pretrained: bool = False,
    strides: list[int] | None = None,
    **kwargs,
) -> BagNet:
    """
    Constructs a Bagnet-9 model.

    :param n_classes: number of classes.
    :param logger:
    :param n_color_channels: number of color channels. Defaults to ``3``.
    :param pretrained: If ``True``, returns a model pre-trained on ImageNet.
        Defaults to ``False``.
    :param strides: Strides of the first layer of each residual block.
        Defaults to ``None``.
    :return: A Bagnet-9 model.
    """
    model = BagNet(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes=n_classes,
        logger=logger,
        n_color_channels=n_color_channels,
        strides=strides,
        n_kernel_3x3=2,
        **kwargs,
    )
    if pretrained:
        model = _load_pretrained_model_wights(
            model, "bagnet9", n_color_channels, n_classes
        )
    return model


all_models = {
    "bagnet33": bagnet33,
    "bagnet17": bagnet17,
    "bagnet9": bagnet9,
}

if __name__ == "__main__":
    from icecream import ic
    from torchinfo import summary

    bagnet = bagnet17(3, n_color_channels=3, pretrained=True)

    ic(
        summary(
            bagnet,
            input_data=(3, 224, 224),
            col_names=("input_size", "output_size", "kernel_size"),
            depth=4,
        )
    )
