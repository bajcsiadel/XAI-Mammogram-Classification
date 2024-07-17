import time
from enum import Enum

import hydra
import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import SubsetRandomSampler, DataLoader

from xai_mam.dataset.dataloaders import CustomDataModule
from xai_mam.models.BagNet._model import BagNetBase
from xai_mam.models._base_classes import BaseTrainer
from xai_mam.utils.config.types import Gpu, ModelParameters, Phase
from xai_mam.utils.log import TrainLogger


class BagNetTrainer(BaseTrainer):
    """
    Abstract base class for BagNet trainer.

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
        model: BagNetBase,
        phases: dict[str, Phase],
        params: ModelParameters,
        loss,  # xai_mam.models.BagNet.config.BagNetLoss,
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
            loss,
            gpu,
            logger,
        )
        self.__criterion = torch.nn.CrossEntropyLoss().to(gpu.device_instance)

        self.logger.info("batch size:")
        with self.logger.increase_indent_context():
            self.logger.info(f"train: {self._phases['main'].batch_size.train}")
            self.logger.info(
                f"validation: {self._phases['main'].batch_size.validation}"
            )

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

    def compute_loss(self, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute the total loss of the model.

        :param kwargs: parameters needed to compute the loss components
        :return:
        """
        cross_entropy = self.__criterion(kwargs["predicted"], kwargs["target"])
        return {"cross_entropy": cross_entropy, "total": cross_entropy}

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
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
        losses = AverageMeter("Loss", ":.4e", Summary.NONE)
        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

        totals = None
        n_batches = 0
        true_labels = np.array([])
        predicted_labels = np.array([])

        start = time.time()
        grad_req = torch.enable_grad() if optimizer is not None else torch.no_grad()

        with grad_req:
            for _, (images, target) in enumerate(dataloader):
                # measure data loading time
                data_time.update(time.time() - start)

                true_labels = np.append(true_labels, target.numpy())

                # move data to the same device as model
                images = images.to(self._gpu.device_instance, non_blocking=True)
                target = target.to(self._gpu.device_instance, non_blocking=True)

                a, b = np.unique(target.cpu().numpy(), return_counts=True)
                self.logger.debug(f"batch sample distribution\n\t{a}\n\t{b}")

                # compute output
                output = self.parallel_model(images)
                loss_values = self._backpropagation(
                    optimizer,
                    predicted=output,
                    target=target,
                )
                _, predicted = torch.max(output.data, 1)

                predicted_labels = np.append(
                    predicted_labels, predicted.cpu().numpy()
                )

                # measure accuracy and record loss
                n_batches += 1
                if totals is None:
                    totals = {k: v.item() for k, v in loss_values.items()}
                else:
                    for k, v in loss_values.items():
                        totals[k] += v.item()

                acc1, acc5 = _accuracy(output, target, topk=(1, 1))
                losses.update(loss_values["total"].item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - start)
                start = time.time()

            with self.logger.increase_indent_context():
                self.logger.info(losses)
                self.logger.info(top1)
                self.logger.info(top5)
                self.logger.info(batch_time)
                self.logger.info("-" * 15)
                metrics = self.compute_metrics(true_labels, predicted_labels, log=True)
                self.logger.info("-" * 15)
                loss_parts = self.compute_loss_parts(totals, n_batches, log=True)

            self.logger.csv_log_values(
                "train_model",
                batch_time.sum,
                totals["cross_entropy"],
                totals["total"],
                *metrics.values(),
            )

            if epoch is not None:
                phase = "train" if optimizer is not None else "eval"
                self.logger.tensorboard.add_scalar(
                    f"accuracy_top1/{phase}", top1.avg, epoch
                )
                self.logger.tensorboard.add_scalars(
                    "accuracy_top1", {f"accuracy_top1/{phase}": top1.avg}
                )
                self.logger.tensorboard.add_scalar(
                    f"accuracy_top5/{phase}", top5.avg, epoch
                )
                self.logger.tensorboard.add_scalars(
                    "accuracy_top5", {f"accuracy_top5/{phase}": top5.avg}
                )

                self.logger.tensorboard.add_scalars(f"loss/{phase}", loss_parts, epoch)
                self.logger.tensorboard.add_scalars(
                    "loss", {f"loss/{phase}": loss_parts["total"]}, epoch
                )
        return top1.avg

    def _get_train_optimizer(self) -> tuple[
        Optimizer, torch.optim.lr_scheduler.LRScheduler
    ]:
        """
        Get the optimizer and learning rate scheduler used in the main phase.

        :return: the optimizer along with the learning scheduler
        """
        optimizer = hydra.utils.instantiate(
            self._phases["main"].optimizer,
            self.model.parameters(),
            self._phases["main"].learning_rates["params"],
            momentum=0.9,
            weight_decay=self._phases["main"].weight_decay,
        )
        lr_scheduler = hydra.utils.instantiate(
            self._phases["main"].scheduler, optimizer=optimizer
        )

        return optimizer, lr_scheduler

    def execute(self, **kwargs):
        """
        Perform the specified phases to train the model.

        :param kwargs: keyword arguments
        """
        train_loader = self._data_module.train_dataloader(
            sampler=self._train_sampler,
            batch_size=self._phases["main"].batch_size.train,
        )
        validation_loader = self._data_module.validation_dataloader(
            sampler=self._validation_sampler,
            batch_size=self._phases["main"].batch_size.validation,
        )

        if self._fold == 1:
            self.logger.log_image_examples(
                self.model,
                train_loader.dataset,
                "train",
                device=self._gpu.device_instance,
            )

        optimizer, lr_scheduler = self._get_train_optimizer()

        for epoch in np.arange(self._phases["main"].epochs) + 1:
            self._epoch += 1
            self.logger.info(f"epoch: \t{epoch} / {self._phases['main'].epochs}")
            self.logger.increase_indent()

            self.train(train_loader, optimizer, epoch)
            accu = self.eval(validation_loader, epoch)

            lr_scheduler.step()

            self.logger.save_model_w_condition(
                state={
                    "state_dict": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "accu": accu,
                },
                model_name=self.model_name(f"{self._epoch}"),
                accu=accu,
            )
            self.logger.decrease_indent()

        self.test()


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.2f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {value" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def _accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
