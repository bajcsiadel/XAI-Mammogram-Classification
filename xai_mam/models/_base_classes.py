import copy
from abc import ABC, abstractmethod
from typing import final

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
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
            self._parallel_model = torch.nn.DataParallel(
                model,
                device_ids=gpu.device_ids,
            )
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
        with self.logger.increase_indent_context():
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
        with self.logger.increase_indent_context():
            self.logger.info("eval")
            self.parallel_model.eval()
            return self._train_and_eval(dataloader, epoch)

    def log_image_examples(self, dataset, set_name="", n_images=8):
        """
        Log some images to the Tensorboard.

        :param dataset:
        :type dataset: xai_mam.dataset.dataloaders.CustomVisionDataset
        :param set_name: name of the subset
        :type set_name: str
        :param n_images: number of images to log. Defaults to ``8``.
        :type n_images: int
        """
        originals = [dataset.get_original(i)[0] for i in range(n_images)]
        transform = A.Compose([
            A.Resize(
                height=dataset.dataset_meta.image_properties.height,
                width=dataset.dataset_meta.image_properties.width,
            ),
            ToTensorV2(),
        ])
        originals = [
            transform(image=image)["image"] for image in originals
        ]
        self.logger.tensorboard.add_image(
            f"{self._data_module.dataset.name} original examples",
            torchvision.utils.make_grid(originals),
        )
        first_batch_input = torch.stack(
            [dataset[i][0] for i in range(n_images * dataset.multiplier)], dim=0
        )
        first_batch_un_normalized = first_batch_input * np.array(
            dataset.dataset_meta.image_properties.std
        )[:, None, None] + np.array(
            dataset.dataset_meta.image_properties.mean
        )[:, None, None]
        self.logger.tensorboard.add_image(
            f"{self._data_module.dataset.name} {set_name} examples (un-normalized)",
            torchvision.utils.make_grid(
                first_batch_un_normalized, nrow=dataset.multiplier
            ),
        )
        self.logger.tensorboard.add_graph(
            self.model, first_batch_input.to(self._gpu.device_instance)
        )
        dataset.reset_used_transforms()

    def log_dataloader_information(self, *dataloaders):
        """
        Log information about the dataloaders. Each dataloader should be represented
        by a tuple where the first element is the name of the dataloader and the
        second is the dataloader itself.

        :param dataloaders:
        :type dataloaders: tuple[str, DataLoader]
        """
        if len(dataloaders) == 0:
            self.logger.info("No dataloaders to log.")
            return

        for name, dataloader in dataloaders:
            self.log_dataset_information(dataloader.dataset, name, dataloader.sampler)

        self.logger.info("batch size:")
        with self.logger.increase_indent_context():
            for name, dataloader in dataloaders:
                self.logger.info(f"{name}: {dataloader.batch_size}")

        self.logger.info("number of batches (dataset length):")
        with self.logger.increase_indent_context():
            for name, dataloader in dataloaders:
                self.logger.info(
                    f"{name}: {len(dataloader)} ({len(dataloader.sampler)})"
                )

    def log_dataset_information(self, dataset, name, sampler=None):
        """
        Log information about the dataset.

        :param dataset:
        :type dataset: xai_mam.dataset.dataloaders.CustomVisionDataset
        :param name: name of the dataset
        :type name: str
        :param sampler: sampler of the dataset. Defaults to ``None``.
        :type sampler: SubsetRandomSampler | numpy.ndarray
        | None
        """
        if sampler is not None:
            if isinstance(sampler, SubsetRandomSampler):
                indices = sampler.indices
            else:
                indices = copy.deepcopy(sampler)
        else:
            indices = np.arange(len(dataset))
        indices = np.unique(indices // dataset.multiplier)
        self.logger.info(f"{name}")
        self.logger.increase_indent()
        self.logger.info(f"size: {len(sampler)} ({len(dataset.targets[indices])} x {dataset.multiplier})")
        self.logger.info("distribution:")
        self.logger.increase_indent()
        distribution = pd.DataFrame(columns=["count", "perc"])
        classes = np.unique(dataset.targets[indices])
        for cls in classes:
            count = np.sum(dataset.targets[indices] == cls) * dataset.multiplier
            distribution.loc[cls] = [count, count / len(sampler)]
        distribution["count"] = distribution["count"].astype("int")
        self.logger.info(f"{distribution.to_string(formatters={'perc': '{:.2%}'.format})}")
        self.logger.decrease_indent(times=2)
