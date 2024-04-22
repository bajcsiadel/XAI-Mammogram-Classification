from abc import abstractmethod
from functools import partial

import numpy as np
import torch.nn as nn
from torchsummary import summary

from xai_mam.models.utils.backbone_features._classes import BackboneFeatureMeta
from xai_mam.models.utils.helpers import get_state_dict

__model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResidualBlock(nn.Module):
    expansion = None
    num_layers = None

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x):
        ...


class BasicBlock(ResidualBlock):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None, **kwargs):
        super(BasicBlock, self).__init__(in_channels, out_channels)
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # if stride is not 1 then self.down_sample cannot be None
        self.down_sample = down_sample
        self.stride = stride
        self.in_channels = in_channels

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        # the residual connection
        out += residual
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class Bottleneck(ResidualBlock):
    # class attribute
    expansion = 4
    num_layers = 3

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        down_sample=None,
    ):
        super(Bottleneck, self).__init__(in_channels, out_channels * self.expansion)
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        if kernel_size == 3:
            make_conv_layer = partial(conv3x3, padding=padding)
        else:
            padding = 0
            make_conv_layer = conv1x1
        self.conv2 = make_conv_layer(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # if stride is not 1 then self.down_sample cannot be None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.down_sample = down_sample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [1, self.kernel_size, 1]
        block_strides = [1, self.stride, 1]
        block_paddings = [0, self.padding, 0]

        return block_kernel_sizes, block_strides, block_paddings


class BaseResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channels,
        channels_per_layer=None,
        kernels=None,
        strides=None,
        paddings=None,
        zero_init_residual=False,
    ):
        super(BaseResNet, self).__init__()

        if channels_per_layer is None:
            channels_per_layer = [2 ** (6 + i) for i in range(len(layers))]
        elif len(channels_per_layer) != len(layers):
            raise ValueError(
                f"channels_per_layer ({len(channels_per_layer)}) does "
                f"not have same length as layers ({len(layers)})"
            )

        if block is Bottleneck:
            if kernels is None:
                kernels = [[3] * num_blocks for num_blocks in layers]
            elif type(kernels) is int:
                kernels = [[kernels] * num_blocks for num_blocks in layers]
            if len(layers) != len(kernels):
                raise ValueError(
                    f"kernels ({len(kernels)}) does not have same "
                    f"length as layers ({len(layers)})"
                )
            else:
                for i in range(len(kernels)):
                    if type(kernels[i]) is int:
                        kernels[i] = [kernels[i]] * layers[i]
            if np.any(np.array(list(map(len, kernels))) != np.array(layers)):
                raise ValueError(
                    f"kernel values not specified for each layer of "
                    f"each block ({np.array(list(map(len, kernels)))} "
                    f"!= {layers})"
                )

            if strides is None:
                strides = [1] + [2] * (len(layers) - 1)
            elif type(strides) is int:
                strides = [strides] * len(layers)
            if len(layers) != len(strides):
                raise ValueError(
                    f"strides ({len(strides)}) does not have same "
                    f"length as layers ({len(layers)})"
                )

            if paddings is None:
                paddings = [1] * len(layers)
            elif type(paddings) is int:
                paddings = [paddings] * len(layers)
            if len(layers) != len(paddings):
                raise ValueError(
                    f"paddings ({len(paddings)}) does not have same "
                    f"length as layers ({len(layers)})"
                )

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers
        self.in_channels = in_channels

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        for i, (channels, num_blocks, kernel_size, stride, padding) in enumerate(
            zip(channels_per_layer, layers, kernels, strides, paddings, strict=True),
            start=1,
        ):
            self.__setattr__(
                f"layer{i}",
                self._make_layer(
                    block=block,
                    channels=channels,
                    num_blocks=num_blocks,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, channels, num_blocks, kernel_size, stride=1, padding=0
    ):
        down_sample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            down_sample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.expansion, stride),
                nn.BatchNorm2d(channels * block.expansion),
            )

        # only the first block has down_sample that is possibly not None
        layers = [
            block(
                self.in_channels,
                channels,
                kernel_size=kernel_size[0],
                stride=stride,
                down_sample=down_sample,
                padding=padding,
            )
        ]

        self.in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, channels, kernel_size[i]))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            (
                block_kernel_sizes,
                block_strides,
                block_paddings,
            ) = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetFeatures(nn.Module):
    """
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    """

    def __init__(
        self,
        block,
        layers,
        color_channels=3,
        channels=64,
        channels_per_layer=None,
        kernels=None,
        strides=None,
        paddings=None,
        zero_init_residual=False,
    ):
        super(ResNetFeatures, self).__init__()

        # stem
        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(
            color_channels,
            channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        self.residual_blocks = BaseResNet(
            block,
            layers,
            channels,
            channels_per_layer,
            kernels,
            strides,
            paddings,
            zero_init_residual,
        )

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.residual_blocks(x)

        return x

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        """
        the number of conv layers in the network, not counting the number
        of bypass layers
        """
        return (
            self.block.num_layers * self.layers[0]
            + self.block.num_layers * self.layers[1]
            + self.block.num_layers * self.layers[2]
            + self.block.num_layers * self.layers[3]
            + 1
        )


def resnet18_features(color_channels=3, pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param pretrained: If ``True``, returns a model pretrained on ImageNet.
        Defaults to ``False``.
    :type pretrained: bool
    :return: A ResNet-18 model
    :rtype: ResNetFeatures
    """
    model = ResNetFeatures(
        BasicBlock, [2, 2, 2, 2], color_channels=color_channels, **kwargs
    )
    if pretrained:
        pretrained_state_dict = get_state_dict(__model_urls["resnet18"], color_channels)
        model.load_state_dict(pretrained_state_dict, strict=False)
    return model


def resnet20_features(color_channels=3, pretrained=False, **kwargs):
    """
    Constructs a ResNet-20 model.

    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param pretrained: If ``True``, returns a model pretrained on ImageNet.
        Defaults to ``False``.
    :type pretrained: bool
    :return: A ResNet-20 model
    :rtype: ResNetFeatures
    """
    model = ResNetFeatures(
        BasicBlock, [3, 3, 3], color_channels=color_channels, channels=16, **kwargs
    )
    return model


def resnet34_features(color_channels=3, pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.

    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param pretrained: If ``True``, returns a model pretrained on ImageNet.
        Defaults to ``False``.
    :type pretrained: bool
    :return: A ResNet-34 model
    :rtype: ResNetFeatures
    """
    model = ResNetFeatures(
        BasicBlock, [3, 4, 6, 3], color_channels=color_channels, **kwargs
    )
    if pretrained:
        pretrained_state_dict = get_state_dict(__model_urls["resnet34"], color_channels)
        model.load_state_dict(pretrained_state_dict, strict=False)
    return model


def resnet50_features(color_channels=3, pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.

    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param pretrained: If ``True``, returns a model pretrained on ImageNet.
        Defaults to ``False``.
    :type pretrained: bool
    :return: A ResNet-50 model
    :rtype: ResNetFeatures
    """
    model = ResNetFeatures(
        Bottleneck, [3, 4, 6, 3], color_channels=color_channels, **kwargs
    )
    if pretrained:
        pretrained_state_dict = get_state_dict(__model_urls["resnet50"], color_channels)
        model.load_state_dict(pretrained_state_dict, strict=False)
    return model


def resnet101_features(color_channels=3, pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.

    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param pretrained: If ``True``, returns a model pretrained on ImageNet.
        Defaults to ``False``.
    :type pretrained: bool
    :return: A ResNet-101 model
    :rtype: ResNetFeatures
    """
    model = ResNetFeatures(
        Bottleneck, [3, 4, 23, 3], color_channels=color_channels, **kwargs
    )
    if pretrained:
        pretrained_state_dict = get_state_dict(
            __model_urls["resnet101"], color_channels
        )
        model.load_state_dict(pretrained_state_dict, strict=False)
    return model


def resnet152_features(color_channels=3, pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.

    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param pretrained: If ``True``, returns a model pretrained on ImageNet.
        Defaults to ``False``.
    :type pretrained: bool
    :return: A ResNet-152 model
    :rtype: ResNetFeatures
    """
    model = ResNetFeatures(
        Bottleneck, [3, 8, 36, 3], color_channels=color_channels, **kwargs
    )
    if pretrained:
        pretrained_state_dict = get_state_dict(
            __model_urls["resnet152"], color_channels
        )
        model.load_state_dict(pretrained_state_dict, strict=False)
    return model


all_features = {
    "resnet18": BackboneFeatureMeta(
        url=__model_urls["resnet18"], construct=resnet18_features
    ),
    "resnet34": BackboneFeatureMeta(
        url=__model_urls["resnet34"], construct=resnet34_features
    ),
    "resnet50": BackboneFeatureMeta(
        url=__model_urls["resnet50"], construct=resnet50_features
    ),
    "resnet101": BackboneFeatureMeta(
        url=__model_urls["resnet101"], construct=resnet101_features
    ),
    "resnet152": BackboneFeatureMeta(
        url=__model_urls["resnet152"], construct=resnet152_features
    ),
}


if __name__ == "__main__":
    from icecream import ic

    # r18_features = resnet18_features(pretrained=True)
    # print(r18_features)
    #
    # r34_features = resnet34_features(pretrained=True)
    # print(r34_features)

    b = resnet50_features()
    ic(
        summary(
            b,
            input_data=(3, 112, 112),
            col_names=("input_size", "output_size", "kernel_size"),
            depth=4,
        )
    )
    # r50_features = resnet50_features(pretrained=True)
    # ic(r50_features)
    # for m in r50_features._modules.items():
    #     ic(m)

    # r101_features = resnet101_features(pretrained=True)
    # print(r101_features)
    #
    # r152_features = resnet152_features(pretrained=True)
    # print(r152_features)
