import datetime
import time
from abc import ABC, abstractmethod
from typing import final

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary

from xai_mam.dataset.dataloaders import CustomDataModule
from xai_mam.utils.config.types import Gpu, ModelParameters, Phase
from xai_mam.utils.log import TrainLogger


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
    :param data_module:
    :param train_sampler: sampler of the training set
    :param validation_sampler: sampler of the validation set
    :param model: model to train
    :param phases: phases of the train process
    :param params: parameters of the model
    :param gpu: gpu properties
    :param logger: logging object
    """

    def __init__(
        self,
        fold: int | None,
        data_module: CustomDataModule,
        train_sampler: SubsetRandomSampler | None,
        validation_sampler: SubsetRandomSampler | None,
        model: nn.Module,
        phases: dict[str, Phase],
        params: ModelParameters,
        gpu: Gpu,
        logger: TrainLogger,
    ):
        if not gpu.disabled:
            self._parallel_model = torch.nn.DataParallel(
                model,
                device_ids=gpu.device_ids,
            )
        else:
            self._parallel_model = model
        self._gpu = gpu
        self._phases = phases
        self._params = params

        self._data_module = data_module
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler

        self._fold = fold
        self._epoch = 0

        self.logger = logger

        if fold == 1:
            logger.info("")
            logger.info(f"{model}\n")

            logger.info("")
            logger.info(
                summary(
                    model,
                    input_size=(
                        data_module.dataset.image_properties.n_color_channels,
                        data_module.dataset.image_properties.height,
                        data_module.dataset.image_properties.width,
                    ),
                    depth=6,
                    batch_dim=0,
                    verbose=0,
                )
            )

    @property
    def model(self) -> torch.nn.Module:
        """
        Returns the core model of the parallel model.

        :return: the core model of the parallel model
        """
        if isinstance(self._parallel_model, torch.nn.DataParallel):
            return self._parallel_model.module

        return self._parallel_model

    @property
    def parallel_model(self) -> torch.nn.DataParallel | torch.nn.Module:
        """
        Returns the parallel model (could be simple model if gpu is disabled).

        :return: model to train
        """
        return self._parallel_model

    def model_name(self, name: str) -> str:
        """
        Concatenate fold number to the output model name.

        :param name: name of the file
        :return: name containing the fold number
        """
        if self._fold is not None:
            name = f"{self._fold}-{name}"
        return name

    @abstractmethod
    def execute(self, **kwargs):
        """
        Perform the specified phases to train the model.

        :param kwargs: keyword arguments
        """
        ...

    @abstractmethod
    def compute_loss(self, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute the total loss of the model.

        :param kwargs: parameters needed to compute the loss components
        :return:
        """
        ...

    def _backpropagation(
        self, optimizer: Optimizer | None, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Perform the backpropagation: computing the loss and if the
        optimizer is specified then push the gradients back to the optimizer.

        :param optimizer:
        :param kwargs: other parameters needed to compute the loss
        :return: components of the loss term
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
        dataloader: DataLoader,
        optimizer: Optimizer = None,
        epoch: int = None,
        **kwargs,
    ):
        """
        Execute train/eval steps of the model.

        :param dataloader:
        :param optimizer: Defaults to ``None``.
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        """
        ...

    def train(
        self,
        dataloader: DataLoader,
        optimizer: Optimizer,
        epoch: int = None,
        **kwargs,
    ) -> float:
        """
        Execute train step of the model.

        :param dataloader:
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :param optimizer: Defaults to ``None``.
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        """
        with self.logger.increase_indent_context():
            self.logger.info("train")
            self.parallel_model.train()
            return self._train_and_eval(dataloader, optimizer, epoch, **kwargs)

    def eval(
        self,
        dataloader: DataLoader,
        epoch: int = None,
        **kwargs,
    ) -> float:
        """
        Execute eval step of the model.

        :param dataloader:
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        """
        with self.logger.increase_indent_context():
            self.logger.info("eval")
            self.parallel_model.eval()
            return self._train_and_eval(dataloader, epoch=epoch)

    def test(self) -> float:
        """
        Evaluate the trained model on the test set.

        :return: accuracy of the model on the test set
        """
        test_loader = self._data_module.test_dataloader(128)

        self.logger.info("start testing")
        start = time.time()
        test_accuracy = self.eval(test_loader)
        self.logger.info(f"test accuracy: {test_accuracy:.2%}")
        self.logger.info(
            f"test ended in {datetime.timedelta(seconds=int(time.time() - start))}"
        )
        self.logger.csv_log_index(
            "train_model", (self._fold, self._epoch, "test")
        )
        return test_accuracy
