import contextlib
import logging
import os
import sys
import traceback
import typing as typ
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from textwrap import indent

import albumentations as A
import numpy as np
import omegaconf
import pandas as pd
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from xai_mam.dataset.dataloaders import CustomVisionDataset, CustomDataModule
from xai_mam.utils import errors
from xai_mam.utils.config.types import FilePrefixes, Outputs


# fmt off
class SpecialCharacters(StrEnum):
    tick = "\u2714"  # noqa
    cross = "\u2718"


# fmt on


class ScriptLogger(logging.Logger):
    """
    Object for managing the log directory

    :param name: The name of the logger
    :param log_location: location where the log file will be saved.
    Defaults to ``None`` ==> hydra output directory
    """

    special_characters = SpecialCharacters

    def __init__(self, name: str, log_location: Path | str | None = None):
        super().__init__(name)
        self.parent = logging.root

        self._log_location = Path(log_location or HydraConfig.get().runtime.output_dir)

        # Ensure the directories exist
        self.log_location.mkdir(parents=True, exist_ok=True)

        self._indent = 0

        logging.getLogger().manager.loggerDict[name] = self
        self.info(f"Log dir: {self.log_location}")

    @contextlib.contextmanager
    def increase_indent_context(self, times: int = 1):
        """
        Context manager increasing the indent,

        :param times:
        """
        self.increase_indent(times)
        yield
        self.decrease_indent(times)

    def increase_indent(self, times: int = 1):
        """
        Increase the indentation

        :param times:
        """
        if times < 0:
            raise ValueError("times must be non-negative")
        self._indent += 4 * times

    def decrease_indent(self, times: int = 1):
        """
        Decrease the indentation.

        :param times:
        """
        if times < 0:
            raise ValueError("times must be non-negative")
        self._indent = max(0, self._indent - 4 * times)

    def print_symbol(self, condition: bool) -> str:
        """
        Get tick or cross based on the given condition.

        :param condition:
        :return: tick if the condition is ``True``, cross otherwise.
        """
        return (
            self.special_characters.tick if condition else self.special_characters.cross
        )

    @property
    def log_location(self):
        return self._log_location

    def _log(
        self,
        level: int,
        msg: str,
        args,
        exc_info=None,
        extra: dict[str, object] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ):
        """
        Write a message to the log file

        :param level: the level of the message, e.g. logging.INFO
        :param msg: the message string to be written to the log file
        :param args: arguments for the message
        :param exc_info: exception info. Defaults to ``None``.
        :param extra: extra information. Defaults to ``None``.
        :param stack_info: whether to include stack info. Defaults to ``False``.
        :param stacklevel: the stack level. Defaults to ``1``.
        """
        if type(msg) is not str:
            msg = str(msg)

        msg = indent(msg, self._indent * " ")

        for line in msg.splitlines():
            super()._log(
                level,
                f"{line}",
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel,
            )

    def exception(self, ex: Exception, warn_only: bool = False, **kwargs):
        """
        Customize logging an exception

        :param ex:
        :param warn_only: Defaults to False
        """
        if warn_only:
            log_fn = self.warning
        else:
            log_fn = self.error
        log_fn(f"{type(ex).__name__}: {ex}", **kwargs)
        log_fn(traceback.format_exc(), **kwargs)

    def __call__(self, message: str):
        """
        Log a message.

        :param message:
        """
        level, msg = message.split(": ", 1)
        match level:
            case "INFO":
                self.info(msg)
            case "WARNING":
                self.warning(msg)
            case "ERROR":
                self.error(msg)
            case _:
                self.log(logging.INFO, message)

    def log_dataset(
        self,
        dataset: CustomVisionDataset,
        name: str = "",
        sampler: SubsetRandomSampler | np.ndarray | None = None
    ):
        """
        Log information about a dataset.

        :param dataset:
        :param name: name of the dataset
        :param sampler: sampler to be applied on the dataset if any.
            Defaults to ``None``.
        """
        if sampler is not None:
            if isinstance(sampler, SubsetRandomSampler):
                indices = sampler.indices
            else:
                indices = copy.deepcopy(sampler)
        else:
            indices = dataset.indices

        original_indices = np.unique(indices // dataset.multiplier)
        self.info(f"{name}")
        self.increase_indent()
        self.info(f"size: {len(indices)} ({len(original_indices)} x {dataset.multiplier})")
        self.info(f"classes to numbers: {dataset.class_to_number}")
        self.info("distribution:")
        self.increase_indent()
        distribution = pd.DataFrame(columns=["count", "perc"])
        classes = np.unique(dataset.targets)
        for cls in classes:
            count = np.sum(dataset.targets[indices] == cls)
            distribution.loc[cls] = [count, count / len(indices)]
        distribution["count"] = distribution["count"].astype("int")
        self.info(
            f"{distribution.to_string(formatters={'perc': '{:3.2%}'.format})}")
        self.decrease_indent(times=2)

    def log_dataloader(self, *dataloaders: tuple[str, DataLoader]):
        """
        Log information about the dataloaders. Each dataloader should be represented
        by a tuple where the first element is the name of the dataloader and the
        second is the dataloader itself.

        :param dataloaders:
        """
        if len(dataloaders) == 0:
            self.info("No dataloaders to log.")
            return

        for name, dataloader in dataloaders:
            self.log_dataset(dataloader.dataset, name, dataloader.sampler)  # noqa

        self.info("batch size:")
        with self.increase_indent_context():
            for name, dataloader in dataloaders:
                self.info(f"{name}: {dataloader.batch_size}")

        self.info("number of batches (dataset length):")
        with self.increase_indent_context():
            for name, dataloader in dataloaders:
                self.info(
                    f"{name}: {len(dataloader)} ({len(dataloader.sampler)})"  # noqa
                )

    def log_data_module(self, datamodule: CustomDataModule):
        """
        Log information about the data module

        :param datamodule:
        """
        self.log_dataset(datamodule.train_data, "train")
        if datamodule.validation_data is not None:
            self.log_dataset(datamodule.validation_data, "validation")
        self.log_dataset(datamodule.push_data, "push")
        self.log_dataset(datamodule.test_data, "test")


class TrainLogger(ScriptLogger):
    """
    Object for managing the log directory

    :param name: The name of the logger
    :param outputs: output dirs and file prefixes
    :param tensorboard: enables TensorBoard logging. Defaults to ``True``.
    :param log_location: location where the log file will be saved.
    Defaults to ``None`` ==> hydra output directory
    """

    def __init__(
        self,
        name: str,
        outputs: Outputs,
        tensorboard: bool = True,
        log_location: Path | str | None = None,
    ):
        super().__init__(name, log_location)

        self.__logs = dict()

        self.__outputs = omegaconf.OmegaConf.to_object(outputs)

        # Ensure the directories exist
        self.metadata_location.mkdir(parents=True, exist_ok=True)
        self.checkpoint_location.mkdir(parents=True, exist_ok=True)

        if tensorboard:
            self.tensorboard_location.mkdir(parents=True, exist_ok=True)
            self.__tensorboard_writer = SummaryWriter(
                log_dir=str(self.tensorboard_location)
            )
        else:
            self.__tensorboard_writer = None

        self.__indent = 0

        logging.getLogger().manager.loggerDict[name] = self

    @property
    def checkpoint_location(self) -> Path:
        return self._log_location / self.__outputs.dirs.checkpoints

    @property
    def image_location(self) -> Path:
        return self._log_location / self.__outputs.dirs.image

    @property
    def metadata_location(self) -> Path:
        return self._log_location / self.__outputs.dirs.metadata

    @property
    def tensorboard_location(self) -> Path:
        return self._log_location / self.__outputs.dirs.tensorboard

    @property
    def file_prefixes(self) -> FilePrefixes:
        """
        :return: file prefixes defined in the config file
        """
        self.debug(str(self.__outputs))
        self.debug(str(self.__outputs.file_prefixes))
        return self.__outputs.file_prefixes

    @property
    def tensorboard(self) -> SummaryWriter:
        return self.__tensorboard_writer

    def log_command_line(self):
        """
        Generate a script that can be used to run the experiment again
        """
        python_file = sys.argv[0]
        params = " ".join(HydraConfig.get().overrides.task)
        screen_name = "mam-ppn-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        bash_script_file = self.metadata_location / "run-experiment.sh"
        with bash_script_file.open(mode="w") as fd:
            fd.write("#!/bin/bash\n")
            fd.write("\n")
            fd.write(f"cd {os.getenv('PROJECT_ROOT')}\n")
            fd.write("\n")
            fd.write("# check if environment exists\n")
            fd.write("poetry env list > /dev/null\n")
            fd.write('if { [ $? -ne 0 ] && [ -f "pyproject.toml" ]; }\n')
            fd.write("then\n")
            fd.write("\tpoetry install\n")
            fd.write("else\n")
            fd.write(
                '\techo "No pyproject.toml found. '
                'Please run this script from the project root."\n'
            )
            fd.write("\texit 1\n")
            fd.write("fi\n")
            fd.write("\n")
            fd.write(f"screen -dmS {screen_name}\n")
            fd.write(
                f"screen -S {screen_name} -X stuff "
                f'"poetry run python {python_file} {params}"\n'
            )
            fd.write("# attaching the screen\n")
            fd.write(f"screen -r {screen_name}\n")

    def create_csv_log(
        self,
        log_name: str,
        key_name: str | list[str],
        *value_names: str,
        exist_ok: bool = False,
    ):
        """
        Create a csv for logging information.

        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key
            (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        :param exist_ok: defines if it is okay to append to result file.
            Defaults to ``False``.
        """
        if log_name in self.__logs.keys():
            if not exist_ok:
                raise FileExistsError("Log already exists!")
            else:
                # the file already exists
                return
        if type(key_name) is str:
            key_name = (key_name,)
        # Add to existing logs
        self.__logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        (self.log_location / f"{log_name}.csv").write_text(
            ",".join(key_name + value_names) + "\n"
        )

    def csv_log_index(self, log_name, key):
        """
        Log the given index in an existent log file

        :param log_name: name of the log file
        :type log_name: str
        :param key: index of the current row
        :type key: str|typ.Iterable[str]
        """
        if log_name not in self.__logs.keys():
            raise FileNotFoundError("Log does not exist!")
        if type(key) is str:
            key = (key,)
        if len(key) != len(self.__logs[log_name][0]):
            raise errors.CsvMismatchedColumnsError("Not all indices are logged!")
        with (self.log_location / f"{log_name}.csv").open(mode="a") as f:
            f.write(",".join(str(v) for v in key) + ",")

    def csv_log_values(self, log_name: str, *values: any):
        """
        Log values in an existent log file. The key should be
        specified in advance by calling create_csv_log

        :param log_name: The name of the log file
        :param values: value attributes that will be stored in the log
        :raises FileNotFoundError: If the log file does not exist
        :raises errors.CsvMismatchedColumnsError: If the number of
        values does not match the number of columns
        """
        if log_name not in self.__logs.keys():
            raise FileNotFoundError("Log does not exist!")
        if len(values) != len(self.__logs[log_name][1]):
            raise errors.CsvMismatchedColumnsError(
                f"Not all required values are logged! "
                f"Expected {len(self.__logs[log_name][1])}, got {len(values)}"
            )
        # Write a new line with the given values
        with (self.log_location / f"{log_name}.csv").open(mode="a") as f:
            f.write(",".join(str(v) for v in values) + "\n")

    def csv_log_line(self, log_name: str, key: str | list[str], *values: any):
        """
        Log the given line in an existent log file

        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        :raises FileNotFoundError: If the log file does not exist
        :raises errors.CsvMismatchedColumnsError: If the number of values
        does not match the number of columns
        """
        self.csv_log_index(log_name, key)
        self.csv_log_values(log_name, *values)

    def save_model(
        self, model_name: str, state: dict[str, any], model_location: Path | str = None
    ):
        """
        Save the model.

        :param model_name: name of the file in which the model is saved
        :param state: state of the training including the model state dict, the
            optimizer state dict and the scheduler state dict, the epoch and
            the accuracy
        :param model_location: location where the model is saved. If it is
            ``None``, then the model will be saved into the ``"checkpoints"``
            folder. Defaults to ``None``.
        :type model_location: pathlib.Path | None
        """
        model_location = model_location or self.checkpoint_location
        torch.save(
            obj=state,
            f=model_location / f"{model_name}.pth",
        )

    def save_model_w_condition(
        self,
        model_name: str,
        state: dict[str, any],
        accu: float,
        target_accu: float = 0.6,
        model_location: Path | str = None,
    ):
        """
        Save the model if it's accuracy reaches a given threshold.

        :param model_name: name of the file in which the model is saved
        :param state: state of the training including the model state dict, the
            optimizer state dict and the scheduler state dict, the epoch and
            the accuracy
        :param accu: test accuracy of the model
        :param target_accu: accuracy threshold. Defaults to ``0.6``.
        :param model_location: location where the model is saved. If it is
            ``None``, then the model will be saved into the ``"checkpoints"``
            folder. Defaults to ``None``.
        """
        if accu > target_accu:
            with self.increase_indent_context():
                self.info(f"above {target_accu:.2%}")
            self.save_model(f"{model_name}-{accu:.4f}", state, model_location)

    def save_image(
        self,
        image_name: Path | str,
        image: np.ndarray,
        image_location: Path | str = None,
    ):
        """
        Save the image.

        :param image_name: name of the output image
        :param image: content of the image to save
        :param image_location: location where the image is saved. Defaults to ``None``.
        """
        if image_location is None:
            image_location = self.image_location

        if type(image_name) is str or (
            type(image_name) is Path and not image_name.is_absolute()
        ):
            image_name = image_location / image_name

        if image.max() > 1:
            image = image / 255.0
        if image.shape[-1] == 1:
            plt.imsave(
                fname=image_name,
                arr=image.squeeze(axis=2),
                cmap="gray",
            )
        else:
            plt.imsave(
                fname=image_name,
                arr=image,
            )

    def log_image_examples(
        self,
        model: nn.Module,
        dataset: CustomVisionDataset,
        set_name: str = "",
        n_images: int = 8,
        device: str | torch.device = "cpu",
    ):
        """
        Log some images to the Tensorboard.

        :param model: model to be trained
        :param dataset: used dataset
        :param set_name: name of the subset
        :param n_images: number of images to log. Defaults to ``8``.
        :param device: device to which the model is compiled
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
        self.tensorboard.add_image(
            f"{dataset.dataset_meta.name} original examples",
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
        self.tensorboard.add_image(
            f"{dataset.dataset_meta.name} {set_name} examples (un-normalized)",
            torchvision.utils.make_grid(
                first_batch_un_normalized, nrow=dataset.multiplier
            ),
        )
        self.tensorboard.add_graph(
            model, first_batch_input.to(device)
        )
        dataset.reset_used_transforms()

    def __enter__(self):
        """
        Enter the context manager
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the tensorboard writer at the end of the experiment.
        """
        self.__tensorboard_writer.close()
