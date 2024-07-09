import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from xai_mam.models.ProtoPNet._trainer import ProtoPNetTrainer


class BackboneTrainer(ProtoPNetTrainer):
    """
    Trainer class to train a ProtoPNet backbone model.
    """

    def compute_loss(self, **kwargs):
        """
        Compute the total loss of the model.

        :param kwargs: parameters needed to compute the loss components
        :return:
        :rtype: dict[str, torch.Tensor]
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

    def _compute_l1_loss(self, **kwargs):
        """
        Compute the L1 loss for the backbone model.

        :param kwargs:
        :return: l1 loss
        :rtype: torch.Tensor
        """
        return self.model.last_layer.weight.norm(p=1)

    def _train_and_eval(
        self,
        dataloader,
        epoch=None,
        optimizer=None,
        use_l1_mask=True,
        **kwargs,
    ):
        """
        Execute train/test steps of the model.

        :param dataloader:
        :type dataloader: torch.utils.data.dataloader.DataLoader
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :type epoch: int | None
        :param optimizer: Defaults to ``None``.
        :type optimizer: torch.optim.Optimizer
        :param use_l1_mask: Defaults to ``True``.
        :type use_l1_mask: bool
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        :rtype: float
        """
        is_train = optimizer is not None
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_cross_entropy = 0

        true_labels = np.array([])
        predicted_labels = np.array([])

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        for image, label in dataloader:
            input_ = image.to(self._gpu.device_instance)
            target_ = label.to(self._gpu.device_instance)
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
                total_cross_entropy += loss_values["cross_entropy"].item()

            predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())

            del input_
            del target_
            del output
            del predicted

        end = time.time()

        total_time = end - start
        cross_entropy = total_cross_entropy / n_batches
        accuracy = n_correct / n_examples
        micro_f1 = f1_score(true_labels, predicted_labels, average="micro")
        macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
        l1_norm = self.model.last_layer.weight.norm(p=1).item()

        with self.logger.increase_indent_context():
            self.logger.info(f"{'time: ':<13}{total_time}")
            self.logger.info(f"{'cross ent: ':<13}{cross_entropy}")
            self.logger.info(f"{'accu: ':<13}{accuracy:.2%}")
            self.logger.info(f"{'micro f1: ':<13}{micro_f1:.2%}")
            self.logger.info(f"{'macro f1: ':<13}{macro_f1:.2%}")
            self.logger.info(f"{'l1: ':<13}{l1_norm}")

        if hasattr(self.logger, "csv_log_values"):
            self.logger.csv_log_values(
                "train_model",
                total_time,
                cross_entropy,
                accuracy,
                micro_f1,
                macro_f1,
                l1_norm,
            )

            if epoch is not None:
                phase = "train" if is_train else "eval"
                self.logger.tensorboard.add_scalar(f"accuracy/{phase}", accuracy, epoch)
                self.logger.tensorboard.add_scalars(
                    "accuracy", {f"accuracy/{phase}": accuracy}, epoch
                )

                write_loss = {
                    f"cross_entropy": cross_entropy
                    * self._loss.coefficients.get("cross_entropy", 1),
                    f"l1": l1_norm * self._loss.coefficients.get("l1", 1e-4),
                    "loss": loss_values["total"].item(),
                }
                self.logger.tensorboard.add_scalars(f"loss/{phase}", write_loss, epoch)
                self.logger.tensorboard.add_scalars(
                    "loss", {f"loss/{phase}": write_loss["loss"]}, epoch
                )

                if "lr" in kwargs:
                    self.logger.tensorboard.add_scalars("lr", kwargs["lr"], epoch)

        return accuracy

    def joint(self):
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
            self.logger.info(f"epoch: \t{epoch} ({self._epoch})")
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
