import datetime
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import SubsetRandomSampler, DataLoader

from xai_mam.dataset.dataloaders import CustomDataModule
from xai_mam.models.ProtoPNet._trainer import ProtoPNetTrainer
from xai_mam.models.ProtoPNet.config import ProtoPNetLoss
from xai_mam.utils.config.types import Gpu, ModelParameters, Phase
from xai_mam.utils.log import TrainLogger


class BackboneTrainer(ProtoPNetTrainer):
    """
    Trainer class to train a ProtoPNet backbone model.

    :param fold: current fold number
    :param data_module:
    :param train_sampler:
    :param validation_sampler:
    :param model: model to train
    :param phases: phases of the train process
    :param params: parameters of the model
    :param loss: loss parameters
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
        model: nn.Module,
        phases: dict[str, Phase],
        params: ModelParameters,
        loss: ProtoPNetLoss,
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

        self._loss = loss

    def compute_loss(self, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute the total loss of the model.

        :param kwargs: parameters needed to compute the loss components
        :return:
        """
        # compute losses
        cross_entropy = self._compute_cross_entropy(**kwargs)
        l1 = self._compute_l1_loss()

        # multiply with coefficient
        cross_entropy = (
            int(self._loss.coefficients.get("cross_entropy", 1)) * cross_entropy
        )
        l1 = self._loss.coefficients.get("l1", 1e-4) * l1

        # compute the total loss
        loss = cross_entropy + l1

        return {
            "cross_entropy": cross_entropy,
            "l1": l1,
            "total": loss,
        }

    def _compute_l1_loss(self, **kwargs) -> torch.Tensor:
        """
        Compute the L1 loss for the backbone model.

        :param kwargs:
        :return: l1 loss
        """
        return self.model.last_layer.weight.norm(p=1)

    def _train_and_eval(
        self,
        dataloader: DataLoader,
        optimizer: Optimizer = None,
        epoch: int = None,
        use_l1_mask: bool = True,
        **kwargs,
    ) -> float:
        """
        Execute train/test steps of the model.

        :param dataloader:
        :param optimizer: Defaults to ``None``.
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :param use_l1_mask: Defaults to ``True``.
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        """
        is_train = optimizer is not None
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        totals = None

        true_labels = np.array([])
        predicted_labels = np.array([])

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        for image, label in dataloader:
            input_ = image.to(self._gpu.device_instance)
            target_ = label.to(self._gpu.device_instance)
            a, b = np.unique(target_.cpu().numpy(), return_counts=True)
            self.logger.debug(f"batch sample distribution\n\t{a}\n\t{b}")
            true_labels = np.append(true_labels, label.numpy())
            with grad_req:
                # nn.Module has implemented __call__() function
                # so no need to call .forward
                output = self.parallel_model(input_)

                # evaluation statistics
                _, predicted = torch.max(output.data, 1)
                n_examples += target_.size(0)
                n_correct += (predicted == target_).sum().item()

                loss_values = self._backpropagation(
                    optimizer, expected=target_, predicted=output
                )

                n_batches += 1
                if totals is None:
                    totals = {k: v.item() for k, v in loss_values.items()}
                else:
                    for k, v in loss_values.items():
                        totals[k] += v.item()

            predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())

            del input_
            del target_
            del output
            del predicted

        end = time.time()

        total_time = end - start
        accuracy = n_correct / n_examples
        l1_norm = self.model.last_layer.weight.norm(p=1).item()

        with self.logger.increase_indent_context():
            self.logger.info(f"{'time: ':<13}{total_time}")
            self.logger.info(f"{'accu: ':<13}{accuracy:.2%}")
            self.logger.info("-" * 15)
            metrics = self.compute_metrics(true_labels, predicted_labels, log=True)
            self.logger.info("-" * 15)
            loss_parts = self.compute_loss_parts(totals, n_batches, log=True)
            self.logger.info(f"{'l1: ':<13}{l1_norm}")

        if hasattr(self.logger, "csv_log_values"):
            self.logger.csv_log_values(
                "train_model",
                total_time,
                totals["cross_entropy"],
                *metrics.values(),
                l1_norm,
            )

            if epoch is not None:
                phase = "train" if is_train else "eval"
                self.logger.tensorboard.add_scalar(f"accuracy/{phase}", accuracy, epoch)
                self.logger.tensorboard.add_scalars(
                    "accuracy", {f"accuracy/{phase}": accuracy}, epoch
                )

                self.logger.tensorboard.add_scalars(f"loss/{phase}", loss_parts, epoch)
                self.logger.tensorboard.add_scalars(
                    "loss", {f"loss/{phase}": loss_parts["total"]}, epoch
                )

                if "lr" in kwargs:
                    self.logger.tensorboard.add_scalars("lr", kwargs["lr"], epoch)

        return accuracy

    def joint(self):
        """
        Perform joint phases of training.
        """
        self._joint()
        self.logger.increase_indent()

        train_loader = self._data_module.train_dataloader(
            sampler=self._train_sampler,
            batch_size=self._phases["joint"].batch_size.train,
        )
        validation_loader = self._data_module.validation_dataloader(
            sampler=self._validation_sampler,
            batch_size=self._phases["joint"].batch_size.validation,
        )

        if self._fold == 1:
            self.logger.log_image_examples(
                self.model,
                train_loader.dataset,
                "train",
                device=self._gpu.device_instance,
            )

        self.logger.log_dataloader(
            ("train", train_loader),
            ("validation", validation_loader)
        )

        joint_optimizer, joint_lr_scheduler = self._get_joint_optimizer()

        for epoch in np.arange(self._phases["joint"].epochs) + 1:
            self._epoch += 1
            self.logger.info(f"epoch: \t{epoch} / {self._phases['joint'].epochs}")
            if self._epoch > 1:
                joint_lr_scheduler.step()

            self.logger.csv_log_index("train_model", (self._fold, self._epoch, "train"))
            _ = self.train(
                dataloader=train_loader,
                optimizer=joint_optimizer,
                epoch=self._epoch,
                lr={
                    k: v
                    for k, v in zip(
                        self._phases["joint"].learning_rates.keys(),
                        joint_lr_scheduler.get_last_lr(),
                        strict=True
                    )
                },
            )

            self.logger.csv_log_index(
                "train_model", (self._fold, self._epoch, "validation")
            )
            accu = self.eval(
                dataloader=validation_loader,
                epoch=self._epoch,
            )
            self.logger.save_model_w_condition(
                state={
                    "state_dict": self.model.state_dict(),
                    "optimizer": joint_optimizer.state_dict(),
                    "scheduler": joint_lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "accu": accu,
                },
                model_name=self.model_name(f"{self._epoch}-backbone"),
                accu=accu,
            )

    def execute(self, **kwargs) -> float:
        """
        Perform the specified phases to train the model.

        :param kwargs: keyword arguments
        :returns: test accuracy of the model
        """
        self.joint()
        return self.test()
