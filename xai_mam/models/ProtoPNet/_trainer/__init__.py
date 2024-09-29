from abc import abstractmethod

import hydra
import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler

from xai_mam.dataset.dataloaders import CustomDataModule
from xai_mam.models.ProtoPNet._model import ProtoPNetBase
from xai_mam.models._base_classes import BaseTrainer
from xai_mam.utils.config.types import Gpu, ModelParameters, Phase
from xai_mam.utils.log import TrainLogger


class ProtoPNetTrainer(BaseTrainer):
    """
    Abstract base class for ProtoPNet trainer.

    :param fold: current fold number
    :param data_module:
    :param train_sampler:
    :param validation_sampler:
    :param model: model to train
    :param phases: phases of the train process
    :param params: parameters of the model
    :param loss: parameters of the loss
    :param gpu: gpu properties
    :param model_initialization_parameters: parameters used to create the model.
        It is saved into the state for reproduction.
    :param logger:
    """

    def __init__(
        self,
        fold: int | None,
        data_module: CustomDataModule,
        train_sampler: SubsetRandomSampler | None,
        validation_sampler: SubsetRandomSampler | None,
        model: ProtoPNetBase,
        phases: dict[str, Phase],
        params: ModelParameters,
        loss,
        gpu: Gpu,
        model_initialization_parameters: dict,
        logger: TrainLogger,
    ):
        super().__init__(
            fold,
            data_module,
            train_sampler,
            validation_sampler,
            model,
            phases,
            params,
            loss,
            gpu,
            model_initialization_parameters,
            logger,
        )

    @abstractmethod
    def _compute_l1_loss(self, **kwargs) -> torch.Tensor:
        """
        Compute the L1 loss for the model.

        :param kwargs:
        :return: l1 loss
        """
        ...

    def _compute_cross_entropy(
        self, predicted: torch.Tensor, expected: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute the cross entropy loss for the model.

        :param predicted: the predicted labels
        :param expected: the expected labels (ground truth)
        :param kwargs: other parameters. If binary cross entropy is needed,
            then ``n_classes`` should be specified.
        :return: cross entropy loss
        """
        if self._loss.binary_cross_entropy:
            one_hot_target = torch.nn.functional.one_hot(expected, kwargs["n_classes"])
            return torch.nn.functional.binary_cross_entropy_with_logits(
                predicted, one_hot_target.float(), reduction="sum"
            )

        return torch.nn.functional.cross_entropy(predicted, expected)

    def _get_joint_optimizer(self) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
    ]:
        """
        Get the optimizer and learning rate scheduler used in the joint phase.

        :return: the optimizer along with the learning scheduler
        """
        joint_optimizer_specs = [
            {
                "params": self.model.features.parameters(),
                "lr": self._phases["joint"].learning_rates["features"],
                "weight_decay": self._phases["joint"].weight_decay,
            },  # bias are now also being regularized
            {
                "params": self.model.add_on_layers.parameters(),
                "lr": self._phases["joint"].learning_rates["add_on_layers"],
                "weight_decay": self._phases["joint"].weight_decay,
            },
        ]
        if not self.model.backbone_only:
            joint_optimizer_specs += [
                {
                    "params": self.model.prototype_vectors,
                    "lr": self._phases["joint"].learning_rates["prototype_vectors"],
                },
            ]

        joint_optimizer = hydra.utils.instantiate(
            self._phases["joint"].optimizer,
            joint_optimizer_specs,
        )
        joint_lr_scheduler = hydra.utils.instantiate(
            self._phases["joint"].scheduler, joint_optimizer
        )

        return joint_optimizer, joint_lr_scheduler

    def _joint(self):
        """
        Prepare model for the joint phase of the training.
        """
        for p in self.model.features.parameters():
            p.requires_grad = True
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = True
        for p in self.model.last_layer.parameters():
            p.requires_grad = True

        self.logger.info("joint")

    @abstractmethod
    def joint(self):
        """
        Perform joint phase of the training.
        """
        ...
