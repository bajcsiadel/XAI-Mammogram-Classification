from abc import ABC, abstractmethod

from xai_mam.models._base_classes import Model


class BagNetBase(Model, ABC):
    """
    Base class for BagNet and BagNetBackbone models.

    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    """

    def __init__(self, n_classes, logger, color_channels=3):
        super().__init__()
        self.in_channels = color_channels
        self.out_channels = n_classes
        self.logger = logger

    @abstractmethod
    def forward(self, x):
        ...
