import copy
import datetime
import time
from functools import partial

import hydra.utils
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import SubsetRandomSampler, DataLoader

from xai_mam.dataset.dataloaders import CustomDataModule
from xai_mam.models.ProtoPNet._helpers import list_of_distances, push
from xai_mam.models.ProtoPNet._trainer import ProtoPNetTrainer
from xai_mam.models.ProtoPNet.config.explainable import ProtoPNetExplainableLoss
from xai_mam.utils.config.types import Gpu, ModelParameters, Phase
from xai_mam.utils.log import TrainLogger
from xai_mam.utils.preprocess import preprocess


class ExplainableTrainer(ProtoPNetTrainer):
    """
    Trainer class to train an explainable ProtoPNet model.

    :param fold: current fold number
    :param data_module:
    :param train_sampler:
    :param validation_sampler:
    :param model: model to train
    :param phases: phases of the train process
    :param params: parameters of the model
    :param loss: loss parameters
    :param gpu: gpu properties
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
        loss: ProtoPNetExplainableLoss,
        gpu: Gpu,
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
            gpu,
            logger,
        )

        self._loss = loss
        preprocess_parameters = {
            "mean": data_module.dataset.image_properties.mean,
            "std": data_module.dataset.image_properties.std,
            "number_of_channels": data_module.dataset.image_properties.n_color_channels,
        }
        self.__preprocess_prototype_fn = partial(preprocess, **preprocess_parameters)
        if train_sampler is not None:
            self.__push_sampler = copy.deepcopy(
                np.array(train_sampler.indices)
            ) // data_module.train_data.multiplier
            self.__push_sampler = np.unique(self.__push_sampler)
        else:
            self.__push_sampler = data_module.train_data.indices

    def compute_loss(self, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute the total loss of the model.

        :param kwargs: parameters needed to compute the loss components
        :return:
        """
        # compute losses
        cross_entropy = self._compute_cross_entropy(**kwargs)
        cluster_cost = self._compute_clustering_cost(**kwargs)
        l1 = self._compute_l1_loss(**kwargs)

        if self.model.class_specific:
            separation_cost = self._compute_separation_cost(**kwargs)
            l2 = self._compute_l2_loss()
        else:
            separation_cost = torch.Tensor([0.0])
            l2 = torch.Tensor([0.0])

        # multiply with coefficient
        cross_entropy = (
            int(self._loss.coefficients.get("cross_entropy", 1)) * cross_entropy
        )
        l1 = self._loss.coefficients.get("l1", 1e-4) * l1
        l2 = self._loss.coefficients.get("l2", 0) * l2

        # compute the total loss
        loss = (
            cross_entropy
            + self._loss.coefficients.get("clustering", 8e-1) * cluster_cost
            - self._loss.coefficients.get("separation", 8e-2) * separation_cost
            + l1
            + l2
        )

        return {
            "cross_entropy": cross_entropy,
            "cluster_cost": cluster_cost,
            "separation_cost": separation_cost,
            "l1": l1,
            "l2": l2,
            "total": loss,
        }

    def _compute_l1_loss(self, use_l1_mask: bool = True, **kwargs) -> torch.Tensor:
        """
        Compute the L1 loss for the explainable model.

        :param use_l1_mask: Defaults to ``True``.
        :param kwargs:
        :return: l1 loss
        """
        if self.model.class_specific and use_l1_mask:
            l1_mask = 1 - torch.t(self.model.prototype_class_identity).to(
                self._gpu.device_instance
            )
            return (self.model.last_layer.weight * l1_mask).norm(p=1)

        return self.model.last_layer.weight.norm(p=1)

    def _compute_l2_loss(self) -> torch.Tensor:
        """
        Compute the L2 loss for the explainable model.

        :return: L2 loss
        """
        if self._loss.separation_type == "avg" and self.model.class_specific:
            return (
                torch.mm(
                    self.model.prototype_vectors[:, :, 0, 0],
                    self.model.prototype_vectors[:, :, 0, 0].t(),
                )
                - torch.eye(self.model.n_prototypes).to(self._gpu.device_instance)
            ).norm(p=2)

        return torch.Tensor(0.0)

    def _compute_clustering_cost(
        self, expected: torch.Tensor, min_distances: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute the clustering cost.

        :param expected: the expected label (ground truth)
        :param min_distances:
        :param kwargs:
        :return: clustering cost
        """
        if self.model.class_specific:
            max_dist = np.prod(self.model.prototype_shape[1:])

            # prototypes_of_correct_class is a tensor
            # of shape batch_size * num_prototypes
            # calculate cluster cost
            expected = expected.to(self.model.prototype_class_identity.device)
            prototypes_of_correct_class = torch.t(
                self.model.prototype_class_identity[:, expected]
            ).to(self._gpu.device_instance)
            inverted_distances, target_proto_index = torch.max(
                (max_dist - min_distances) * prototypes_of_correct_class,
                dim=1,
            )
            return torch.mean(max_dist - inverted_distances)

        min_distance = torch.min(min_distances, dim=1)
        return torch.mean(min_distance)

    def _compute_separation_cost(
        self, expected: torch.Tensor, min_distances: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute the separation cost for the explainable model.

        :param expected: the expected label (ground truth)
        :param min_distances:
        :param kwargs:
        :return:
        """
        if self.model.class_specific:
            max_dist = np.prod(self.model.prototype_shape[1:])
            expected = expected.to(self.model.prototype_class_identity.device)
            prototypes_of_correct_class = torch.t(
                self.model.prototype_class_identity[:, expected]
            ).to(self._gpu.device_instance)

            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

            match self._loss.separation_type:
                case "max":
                    (
                        inverted_distances_to_nontarget_prototypes,
                        _,
                    ) = torch.max(
                        (max_dist - min_distances) * prototypes_of_wrong_class,
                        dim=1,
                    )
                    return torch.mean(
                        max_dist - inverted_distances_to_nontarget_prototypes
                    )
                case "avg":
                    input_ = kwargs["input_"]
                    min_distances_detached_prototype_vectors = (
                        self.model.prototype_min_distances(
                            input_, detach_prototypes=True
                        )[0]
                    )
                    # calculate avg cluster cost
                    avg_separation_cost = torch.sum(
                        min_distances_detached_prototype_vectors
                        * prototypes_of_wrong_class,
                        dim=1,
                    ) / torch.sum(prototypes_of_wrong_class, dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)

                    return avg_separation_cost
                case "margin":
                    # For each input get the distance
                    # to the closest target class prototype
                    inverted_distances, target_proto_index = torch.max(
                        (max_dist - min_distances) * prototypes_of_correct_class,
                        dim=1,
                    )
                    min_distance_target = max_dist - inverted_distances.reshape((-1, 1))

                    all_distances = kwargs["distances"]
                    min_indices = kwargs["min_indices"]

                    anchor_index = min_indices[
                        torch.arange(0, target_proto_index.size(0), dtype=torch.long),
                        target_proto_index,
                    ].squeeze()
                    all_distances = all_distances.view(
                        all_distances.size(0), all_distances.size(1), -1
                    )
                    distance_at_anchor = all_distances[
                        torch.arange(0, all_distances.size(0), dtype=torch.long),
                        :,
                        anchor_index,
                    ]

                    # For each non-target prototype
                    # compute difference compared to the closest target prototype
                    # d(a, p) - d(a, n) term from TripletMarginLoss
                    distance_pos_neg = (
                        min_distance_target - distance_at_anchor
                    ) * prototypes_of_wrong_class
                    # Separation cost is the margin loss
                    # max(d(a, p) - d(a, n) + margin, 0)
                    return torch.mean(
                        torch.maximum(
                            distance_pos_neg
                            + self._loss.coefficients.get("separation_margin", 0),
                            torch.tensor(0.0, device=distance_pos_neg.device),
                        )
                    )
                case _:
                    raise ValueError(
                        f"separation_type has to be one of "
                        f"[max, mean, margin], got "
                        f"{self._loss.separation_type}"
                    )

        return torch.Tensor(0.0)

    def _train_and_eval(
        self,
        dataloader: DataLoader,
        optimizer: Optimizer = None,
        epoch: int = None,
        use_l1_mask: bool = True,
        **kwargs,
    ) -> float:
        """
        Execute train/eval steps of the model.

        :param dataloader:
        :param optimizer: Defaults to ``None``.
        :param epoch: current step needed for Tensorboard logging. Defaults to ``None``.
        :param kwargs: other parameters
        :return: accuracy achieved in the current step
        """
        is_train = optimizer is not None
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_cross_entropy = 0
        total_cluster_cost = 0
        # separation cost is meaningful only for class_specific
        total_separation_cost = 0

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
                output, additional_out = self.parallel_model(input_)

                # evaluation statistics
                _, predicted = torch.max(output.data, 1)
                n_examples += target_.size(0)
                n_correct += (predicted == target_).sum().item()

                loss_values = self._backpropagation(
                    optimizer,
                    input_=input_,
                    expected=target_,
                    predicted=output,
                    **additional_out._asdict(),  # noqa
                )

                n_batches += 1
                total_cross_entropy += loss_values["cross_entropy"].item()
                total_cluster_cost += loss_values["cluster_cost"].item()
                total_separation_cost += loss_values["separation_cost"].item()

            predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())

            del input_
            del target_
            del output
            del predicted

        end = time.time()

        total_time = end - start
        cross_entropy = total_cross_entropy / n_batches
        cluster_cost = total_cluster_cost / n_batches
        separation_cost = (
            total_separation_cost / n_batches if self.model.class_specific else None
        )
        accuracy = n_correct / n_examples
        micro_f1 = f1_score(true_labels, predicted_labels, average="micro")
        macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
        l1_norm = self.model.last_layer.weight.norm(p=1).item()

        p = self.model.prototype_vectors.view(self.model.n_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p)).item()

        with self.logger.increase_indent_context():
            self.logger.info(f"{'time: ':<13}{total_time}")
            self.logger.info(f"{'cross ent: ':<13}{cross_entropy}")
            self.logger.info(f"{'cluster: ':<13}{cluster_cost}")
            if self.model.class_specific:
                self.logger.info(f"{'separation: ':<13}{separation_cost}")
            self.logger.info(f"{'accu: ':<13}{accuracy:.2%}")
            self.logger.info(f"{'micro f1: ':<13}{micro_f1:.2%}")
            self.logger.info(f"{'macro f1: ':<13}{macro_f1:.2%}")
            self.logger.info(f"{'l1: ':<13}{l1_norm}")
            self.logger.info(f"{'p dist pair: ':<13}{p_avg_pair_dist}")

        self.logger.csv_log_values(
            "train_model",
            total_time,
            cross_entropy,
            cluster_cost,
            separation_cost,
            accuracy,
            micro_f1,
            macro_f1,
            l1_norm,
            p_avg_pair_dist,
        )

        if epoch is not None:
            phase = "train" if is_train else "eval"
            self.logger.tensorboard.add_scalar(f"accuracy/{phase}", accuracy, epoch)
            self.logger.tensorboard.add_scalars(
                "accuracy", {f"accuracy/{phase}": accuracy}, epoch
            )
            write_loss = {
                "cross_entropy": cross_entropy,
                "l1": l1_norm,
                "cluster_cost": cluster_cost,
                "loss": loss_values["total"].item(),
            }

            if self.model.class_specific:
                write_loss["separation_cost"] = separation_cost
            self.logger.tensorboard.add_scalars(f"loss/{phase}", write_loss, epoch)
            self.logger.tensorboard.add_scalars(
                "loss", {f"loss/{phase}": write_loss["loss"]}, epoch
            )

            if "lr" in kwargs:
                self.logger.tensorboard.add_scalars("lr", kwargs["lr"], epoch)

        return accuracy

    def _get_warm_optimizer(self) -> Optimizer:
        """
        Get the optimizer used in the warm-up phase.

        :return: optimizer
        """
        warm_optimizer_specs = [
            {
                "params": self.model.add_on_layers.parameters(),
                "lr": self._phases["warm"].learning_rates["add_on_layers"],
                "weight_decay": self._phases["warm"].weight_decay,
            },
            {
                "params": self.model.prototype_vectors,
                "lr": self._phases["warm"].learning_rates["prototype_vectors"],
            },
        ]
        return hydra.utils.instantiate(
            self._phases["warm"].optimizer,
            warm_optimizer_specs,
        )

    def _get_last_layer_optimizer(self) -> Optimizer:
        """
        Get the optimizer used in fine-tuning phase.

        :return: optimizer
        """
        last_layer_optimizer_specs = [
            {
                "params": self.model.last_layer.parameters(),
                "lr": self._phases["finetune"].learning_rates["classification"],
            }
        ]
        return hydra.utils.instantiate(
            self._phases["finetune"].optimizer,
            last_layer_optimizer_specs,
        )

    def _warm_only(self):
        """
        Prepare model for the warm-up phase of the training.
        """
        for p in self.model.features.parameters():
            p.requires_grad = False
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = True
        self.model.prototype_vectors.requires_grad = True
        for p in self.model.last_layer.parameters():
            p.requires_grad = True

        self.logger.info("warm")

    def _joint(self):
        """
        Prepare model for the joint phase of the training.
        """
        super()._joint()
        self.model.prototype_vectors.requires_grad = True

    def _last_only(self):
        """
        Prepare model for the fine-tuning of the training.
        """
        for p in self.model.features.parameters():
            p.requires_grad = False
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = False
        self.model.prototype_vectors.requires_grad = False
        for p in self.model.last_layer.parameters():
            p.requires_grad = True

        self.logger.info("last layer")

    def warm(self):
        """
        Perform warm-up phase of the training.
        """
        self._warm_only()
        self.logger.increase_indent()

        train_loader = self._data_module.train_dataloader(
            sampler=self._train_sampler,
            batch_size=self._phases["warm"].batch_size.train,
        )
        validation_loader = self._data_module.validation_dataloader(
            sampler=self._validation_sampler,
            batch_size=self._phases["warm"].batch_size.validation,
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

        warm_optimizer = self._get_warm_optimizer()

        for epoch in np.arange(self._phases["warm"].epochs) + 1:
            self._epoch += 1
            self.logger.info(f"warm epoch: \t{epoch} / {self._phases['warm'].epochs}")
            self.logger.increase_indent()

            self.logger.csv_log_index("train_model", (self._fold, epoch, "warm train"))
            _ = self.train(
                dataloader=train_loader,
                optimizer=warm_optimizer,
                epoch=self._epoch,
                lr={
                    k: v
                    for k, v in self._phases["joint"].learning_rates.items()
                },
            )

            self.logger.csv_log_index(
                "train_model", (self._fold, epoch, "warm validation")
            )
            accu = self.eval(
                dataloader=validation_loader,
                epoch=self._epoch,
            )

            self.logger.save_model_w_condition(
                model_name=self.model_name(f"{epoch}-warm"),
                state={
                    "state_dict": self.model.state_dict(),
                    "optimizer": warm_optimizer.state_dict(),
                    "epoch": self._epoch,
                    "accu": accu,
                },
                accu=accu,
            )

            self.logger.decrease_indent()

        self.logger.info("finished warmup")
        self.logger.decrease_indent()

    def joint(self):
        """
        Perform joint phase of the training.
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
        push_loader = self._data_module.push_dataloader(
            sampler=self.__push_sampler,
            batch_size=self._params.push.batch_size,
        )

        if len(self._params.push.push_epochs) == 0:
            self._params.push.define_push_epochs(self._phases["joint"].epochs)
            self.logger.info(f"push epochs: {self._params.push.push_epochs}")

        self.logger.log_dataloader(
            ("train", train_loader),
            ("validation", validation_loader),
            ("push", push_loader)
        )

        joint_optimizer, joint_lr_scheduler = self._get_joint_optimizer()

        for epoch in np.arange(self._phases["joint"].epochs) + 1:
            self._epoch += 1
            self.logger.info(
                f"epoch: \t{epoch} / {self._phases['joint'].epochs} ({self._epoch})"
            )
            self.logger.increase_indent()
            if epoch > 1:
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
                model_name=self.model_name(f"{self._epoch}-no_push"),
                state={
                    "state_dict": self.model.state_dict(),
                    "optimizer": joint_optimizer.state_dict(),
                    "scheduler": joint_lr_scheduler.state_dict(),
                    "epoch": self._epoch,
                    "accu": accu,
                },
                accu=accu,
            )

            if epoch in self._params.push.push_epochs:
                push.push_prototypes(
                    dataloader=push_loader,
                    # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=self.parallel_model,
                    # pytorch network with prototype_vectors
                    class_specific=self.model.class_specific,
                    preprocess_input_function=self.__preprocess_prototype_fn,
                    # normalize
                    prototype_layer_stride=1,
                    # if not None, prototypes will be saved here
                    epoch_number=epoch,
                    # if not provided, prototypes saved previously will be overwritten
                    save_prototype_class_identity=True,
                    logger=self.logger,
                    device=self._gpu.device_instance,
                )

                self.logger.csv_log_index(
                    "train_model", (self._fold, self._epoch, "push validation")
                )
                accu = self.eval(
                    dataloader=validation_loader,
                    epoch=self._epoch,
                )
                self.logger.save_model_w_condition(
                    model_name=self.model_name(f"{self._epoch}-push"),
                    state={
                        "state_dict": self.model.state_dict(),
                        "accu": accu,
                    },
                    accu=accu,
                )

                with self.logger.increase_indent_context():
                    self.last_layer(train_loader, validation_loader)
                    self.logger.info("finished pushing prototypes")
            self.logger.decrease_indent()
        self.logger.info(f"finished training fold {self._fold}")

    def last_layer(self, train_loader: DataLoader, validation_loader: DataLoader):
        """
        Perform fine-tuning.

        :param train_loader:
        :param validation_loader:
        """
        last_layer_optimizer = self._get_last_layer_optimizer()

        if self._params.prototypes.activation_fn != "linear":
            self._last_only()
            with self.logger.increase_indent_context():
                for i in np.arange(self._phases["finetune"].epochs) + 1:
                    self.logger.info(
                        f"iteration:\t{i} / {self._phases['finetune'].epochs}"
                    )

                    self.logger.csv_log_index(
                        "train_model",
                        (self._fold, self._epoch, f"last layer {i} train"),
                    )
                    _ = self.train(
                        dataloader=train_loader,
                        optimizer=last_layer_optimizer,
                    )

                    self.logger.csv_log_index(
                        "train_model",
                        (self._fold, self._epoch, f"last layer {i} validation"),
                    )
                    accu = self.eval(
                        dataloader=validation_loader,
                    )
                    self.logger.save_model_w_condition(
                        model_name=self.model_name(f"{self._epoch}-{i}-push"),
                        state={
                            "state_dict": self.model.state_dict(),
                            "optimizer": last_layer_optimizer.state_dict(),
                            "accu": accu,
                        },
                        accu=accu,
                    )
                self.logger.info("finished fine-tuning last layer")
            # set back to train in joint mode
            self._joint()

    def execute(self, **kwargs):
        """
        Perform the specified phases to train the model.

        :param kwargs: keyword arguments
        """
        for phase in [self.warm, self.joint]:
            start_phase = time.time()
            phase()
            self.logger.info(
                f"{phase.__name__} phase finished in: "
                f"{datetime.timedelta(seconds=int(time.time() - start_phase))}"
            )

        self.test()
