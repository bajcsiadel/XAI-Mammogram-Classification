from abc import ABC, abstractmethod

from xai_mam.models._base_classes import Model


class BagNetBase(Model, ABC):
    """
    Base class for BagNet and BagNetBackbone models.

    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    """

    def __init__(self, n_classes, logger, n_color_channels=3):
        super().__init__()
        self.in_channels = n_color_channels
        self.out_channels = n_classes
        self.logger = logger

        self.logger.create_csv_log(
            "train_model",
            ("fold", "epoch", "phase"),
            "time",
            "cross entropy",
            "loss",
            "accuracy",
            "precision",
            "recall",
            "micro_f1",
            "macro_f1",
            exist_ok=True,
        )

    @abstractmethod
    def forward(self, x):
        ...
