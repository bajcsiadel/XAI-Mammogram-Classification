from torch import nn
from torch.utils import model_zoo

from xai_mam.models._base_classes import Backbone
from xai_mam.models.BagNet._model import BagNetBase
from xai_mam.models.utils.backbone_features import resnet_features
from xai_mam.models.utils.helpers import get_state_dict
from xai_mam.utils.environment import get_env


class BagNetBackbone(BagNetBase, Backbone):
    """
    Corresponding backbone feature of an explainable BagNet model, which is actually
    a ResNet-50.

    :param n_classes: number of classes in the data
    :type n_classes: int
    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    :param n_color_channels: number of color channels in the input. Defaults to ``3``.
    :type n_color_channels: int
    :param pretrained: whether to use a pre-trained model or not. Defaults to ``False``.
    :type pretrained: bool
    """

    def __init__(
        self,
        n_classes,
        logger,
        n_color_channels=3,
        pretrained=False,
    ):
        super(BagNetBackbone, self).__init__(n_classes, logger, n_color_channels)

        self.logger.create_csv_log(
            "train_model",
            ("fold", "epoch", "phase"),
            "time",
            "cross entropy",
            "accuracy",
            "micro_f1",
            "macro_f1",
            "l1",
            exist_ok=True,
        )

        resnet50 = resnet_features.all_features["resnet50"]
        # loads state dict if pretrained is True
        self.features = resnet50.construct(n_color_channels, pretrained)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            self.features.residual_blocks.layer4[-1].out_channels, n_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
