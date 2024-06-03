from abc import abstractmethod

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import hydra
import torch
import torchvision

from xai_mam.dataset.dataloaders import CustomSubset
from xai_mam.models._base_classes import BaseTrainer


class ProtoPNetTrainer(BaseTrainer):
    """
    Abstract base class for ProtoPNet trainer.

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
        model,
        phases,
        params,
        loss,
        gpu,
        logger,
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
            logger,
        )

    def model_name(self, name):
        """
        Concatenate fold number to the output model name.

        :param name: name of the file
        :type name: str
        :return: name containing the fold number
        :rtype: str
        """
        if self._fold is not None:
            name = f"{self._fold}-{name}"
        return name

    @abstractmethod
    def _compute_l1_loss(self, **kwargs):
        """
        Compute the L1 loss for the model.

        :param kwargs:
        :return: l1 loss
        :rtype: torch.Tensor
        """
        ...

    def _compute_cross_entropy(self, predicted, expected, **kwargs):
        """
        Compute the cross entropy loss for the model.

        :param predicted: the predicted labels
        :type predicted: torch.Tensor
        :param expected: the expected labels (ground truth)
        :type expected: torch.Tensor
        :param kwargs: other parameters. If binary cross entropy is needed,
            then ``n_classes`` should be specified.
        :return: cross entropy loss
        :rtype: torch.Tensor
        """
        if self._loss.binary_cross_entropy:
            one_hot_target = torch.nn.functional.one_hot(expected, kwargs["n_classes"])
            return torch.nn.functional.binary_cross_entropy_with_logits(
                predicted, one_hot_target.float(), reduction="sum"
            )

        return torch.nn.functional.cross_entropy(predicted, expected)

    def _get_joint_optimizer(self):
        """
        Get the optimizer and learning rate scheduler used in the joint phase.

        :return: the optimizer along with the learning scheduler
        :rtype: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
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

        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
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

    def execute(self, **kwargs):
        """
        Perform the specified phases to train the model.

        :param kwargs: keyword arguments
        """
        self.joint()

    def log_image_examples(self, dataset, set_name="", n_images=8):
        """
        Log some images to the Tensorboard.

        :param dataset:
        :type dataset: xai_mam.dataset.dataloaders.CustomVisionDataset |
        xai_mam.dataset.dataloaders.CustomSubset
        :param set_name: name of the subset
        :type set_name: str
        :param n_images: number of images to log. Defaults to ``8``.
        :type n_images: int
        """
        if type(dataset) is CustomSubset:
            dataset = dataset.dataset
        originals = [dataset.get_original(i)[0] for i in range(n_images)]
        transform = A.Compose([
            A.Resize(
                height=dataset.dataset_meta.image_properties.height,
                width=dataset.dataset_meta.image_properties.width,
            ),
            ToTensorV2(),
        ])
        originals = [
            transform(image=image)["image"] for image in originals]
        self.logger.tensorboard.add_image(
            f"{self._data_module.dataset.name} original examples",
            torchvision.utils.make_grid(originals),
        )
        first_batch_input = torch.stack(
            [dataset[i][0] for i in range(n_images * dataset.multiplier)], dim=0
        )
        first_batch_un_normalized = first_batch_input * np.array(dataset.dataset_meta.image_properties.std)[:, None, None] + np.array(dataset.dataset_meta.image_properties.mean)[:, None, None]
        self.logger.tensorboard.add_image(
            f"{self._data_module.dataset.name} {set_name} examples (un-normalized)",
            torchvision.utils.make_grid(first_batch_un_normalized, nrow=dataset.multiplier),
        )
        self.logger.tensorboard.add_graph(
            self.model, first_batch_input.to(self._gpu.device)
        )
        dataset.reset_used_transforms()
