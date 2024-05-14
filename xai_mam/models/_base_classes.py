from abc import ABC, abstractmethod
from typing import final

import torch
from torch import nn
from torchinfo import summary


class Model(ABC, nn.Module):
    """
    Base class for all models in the library.
    """

    @property
    @abstractmethod
    def backbone_only(self):
        ...


class Backbone(Model, ABC):
    @property
    @final
    def backbone_only(self):
        return True


class Explainable(Model, ABC):
    @property
    @final
    def backbone_only(self):
        return False


class BaseTrainer(ABC):
    """
    Abstract class for all trainers.

    :param fold: current fold number
    :type fold: int
    :param data_module:
    :type data_module: ProtoPNet.dataset.dataloaders.CustomDataModule
    :param train_sampler:
    :type train_sampler: torch.utils.data.SubsetRandomSampler | None
    :param validation_sampler:
    :type validation_sampler: torch.utils.data.SubsetRandomSampler | None
    :param model: model to train
    :type model: ProtoPNet.models.ProtoPNet._model.ProtoPNetBase
    :param phases: phases of the train process
    :type phases: dict[str, ProtoPNet.utils.config._general_types.network.Phase]
    :param params: parameters of the model
    :type params: ProtoPNet.utils.config._general_types.ModelParameters
    :param loss: loss parameters
    :type loss: ProtoPNet.models.ProtoPNet.config.ProtoPNetLoss
    :param gpu: gpu properties
    :type gpu: ProtoPNet.utils.config.types.Gpu
    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    """

    def __init__(
        self,
        fold,
        data_module,
        train_sampler,
        validation_sampler,
        model: nn.Module,
        phases,
        params,
        loss,
        gpu,
        logger,
    ):
        if not gpu.disabled:
            model = model.to(gpu.device)
            self._parallel_model = torch.nn.DataParallel(model)
        else:
            self._parallel_model = model
        self._gpu = gpu
        self._phases = phases
        self._params = params
        self._loss = loss

        self._data_module = data_module
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler

        self._fold = fold
        self._epoch = 0
        self._step = 0

        self.logger = logger

        if fold == 1:
            logger.info("")
            logger.info(f"{model}\n")

            logger.info("")
            logger.info(
                summary(
                    model,
                    input_size=(
                        data_module.dataset.image_properties.color_channels,
                        data_module.dataset.image_properties.height,
                        data_module.dataset.image_properties.width,
                    ),
                    depth=5,
                    batch_dim=1,
                    device=torch.device(gpu.device),
                    verbose=0,
                )
            )

        data_module.log_data_information(logger)

    @property
    def model(self):
        """
        Returns the core model of the parallel model.

        :return: the core model of the parallel model
        :rtype: torch.nn.Module
        """
        if isinstance(self._parallel_model, torch.nn.DataParallel):
            return self._parallel_model.module

        return self._parallel_model

    @property
    def parallel_model(self):
        """
        Returns the parallel model (could be simple model if gpu is disabled).

        :return: model to train
        :rtype: torch.nn.DataParallel | torch.nn.Module
        """
        return self._parallel_model

    @abstractmethod
    def execute(self, **kwargs):
        """
        Perform the specified phases to train the model.

        :param kwargs: keyword arguments
        """
        ...

    @abstractmethod
    def compute_loss(self, **kwargs):
        """
        Compute the total loss of the model.

        :param kwargs: parameters needed to compute the loss components
        :return:
        :rtype: dict[str, torch.Tensor]
        """
        ...

    def _backpropagation(self, optimizer, **kwargs):
        """
        Perform the backpropagation: computing the loss and if the
        optimizer is specified then push the gradients back to the optimizer.

        :param optimizer:
        :type optimizer: torch.optim.Optimizer
        :param kwargs: other parameters needed to compute the loss
        :return: components of the loss term
        :rtype: dict[str, torch.Tensor]
        """
        loss_values = self.compute_loss(**kwargs)

        if optimizer is not None:
            optimizer.zero_grad()
            loss_values["total"].backward()
            optimizer.step()

        return loss_values

    @abstractmethod
    def _train_and_eval(
        self,
        dataloader,
        epoch=None,
        optimizer=None,
        **kwargs,
    ):
        """
        Execute train/eval steps of the model.

        :param dataloader:
        :type dataloader: torch.utils.data.dataloader.DataLoader
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :type epoch: int | None
        :param optimizer: Defaults to ``None``.
        :type optimizer: torch.optim.Optimizer
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        :rtype: float
        """
        ...

    def train(
        self,
        dataloader,
        epoch=None,
        optimizer=None,
        **kwargs,
    ):
        """
        Execute train step of the model.

        :param dataloader:
        :type dataloader: torch.utils.data.dataloader.DataLoader
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :type epoch: int | None
        :param optimizer: Defaults to ``None``.
        :type optimizer: torch.optim.Optimizer
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        :rtype: float
        """
        self.logger.info("train")
        self.parallel_model.train()
        return self._train_and_eval(dataloader, epoch, optimizer, **kwargs)

    def eval(
        self,
        dataloader,
        epoch=None,
        **kwargs,
    ):
        """
        Execute eval step of the model.

        :param dataloader:
        :type dataloader: torch.utils.data.dataloader.DataLoader
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :type epoch: int | None
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        :rtype: float
        """
        self.logger.info("eval")
        self.parallel_model.eval()
        return self._train_and_eval(dataloader, epoch)
