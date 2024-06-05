import contextlib
import logging
import os
import sys
import traceback
import typing as typ
from datetime import datetime
from pathlib import Path
from textwrap import indent

import omegaconf
import torch
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from xai_mam.utils import errors


class ScriptLogger(logging.Logger):
    """
    Object for managing the log directory

    :param name: The name of the logger
    :type name: str
    """

    def __init__(self, name, log_location=None):
        super().__init__(name)
        self.parent = logging.root

        self._log_location = log_location or Path(HydraConfig.get().runtime.output_dir)

        # Ensure the directories exist
        self.log_location.mkdir(parents=True, exist_ok=True)

        self._indent = 0

        logging.getLogger().manager.loggerDict[name] = self
        self.info(f"Log dir: {self.log_location}")

    @contextlib.contextmanager
    def increase_indent_context(self, times=1):
        self.increase_indent(times)
        yield
        self.decrease_indent(times)

    def increase_indent(self, times=1):
        if times < 0:
            raise ValueError("times must be non-negative")
        self._indent += 4 * times

    def decrease_indent(self, times=1):
        if times < 0:
            raise ValueError("times must be non-negative")
        self._indent = max(0, self._indent - 4 * times)

    @property
    def log_location(self):
        return self._log_location

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        """
        Write a message to the log file

        :param level: the level of the message, e.g. logging.INFO
        :type level: int
        :param msg: the message string to be written to the log file
        :type msg: str
        :param args: arguments for the message
        :type args:
        :param exc_info: exception info. Defaults to None
        :param extra: extra information. Defaults to None
        :type extra: typ.Mapping[str, object] | None
        :param stack_info: whether to include stack info. Defaults to False
        :type stack_info: bool
        :param stacklevel: the stack level. Defaults to 1
        :type stacklevel: int
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

    def exception(self, ex, warn_only=False, **kwargs):
        """
        Customize logging an exception

        :param ex:
        :type ex: Exception
        :param warn_only: Defaults to False
        :type warn_only: bool
        :type ex: Exception
        """
        if warn_only:
            log_fn = self.warning
        else:
            log_fn = self.error
        log_fn(f"{type(ex).__name__}: {ex}", **kwargs)
        log_fn(traceback.format_exc(), **kwargs)

    def __call__(self, message):
        """
        Log a message

        :param message:
        :type message: str
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


class TrainLogger(ScriptLogger):
    """
    Object for managing the log directory

    :param name: The name of the logger
    :type name: str
    :param outputs: output dirs and file prefixes
    :type outputs: xai_mam.utils.config_types.Outputs
    """

    def __init__(self, name, outputs, tensorboard=True, log_location=None):
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
    def checkpoint_location(self):
        return self._log_location / self.__outputs.dirs.checkpoints

    @property
    def image_location(self):
        return self._log_location / self.__outputs.dirs.image

    @property
    def metadata_location(self):
        return self._log_location / self.__outputs.dirs.metadata

    @property
    def tensorboard_location(self):
        return self._log_location / self.__outputs.dirs.tensorboard

    @property
    def file_prefixes(self):
        """
        :return: file prefixes defined in the config file
        :rtype: xai_mam.utils.config._general_types.log.FilePrefixes
        """
        self.debug(str(self.__outputs))
        self.debug(str(self.__outputs.file_prefixes))
        return self.__outputs.file_prefixes

    @property
    def tensorboard(self):
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

    def __call__(self, message):
        """
        Log a message

        :param message:
        :type message: str
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

    def create_csv_log(self, log_name, key_name, *value_names, exist_ok=False):
        """
        Create a csv for logging information.

        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :type log_name: str
        :param key_name: The name of the attribute that is used as key
            (e.g. epoch number)
        :type key_name: str|typing.Iterable[str]
        :param value_names: The names of the attributes that are logged
        :type value_names: str
        :param exist_ok: defines if it is okay to append to result file.
            Defaults to ``False``.
        :type exist_ok: bool
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

    def csv_log_values(self, log_name, *values):
        """
        Log values in an existent log file. The key should be
        specified in advance by calling create_csv_log

        :param log_name: The name of the log file
        :type log_name: str
        :param values: value attributes that will be stored in the log
        :type values: typ.Any
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

    def csv_log_line(self, log_name, key, *values):
        """
        Log the given line in an existent log file

        :param log_name: The name of the log file
        :type log_name: str
        :param key: The key attribute for logging these values
        :type key: str|typ.Iterable[str]
        :param values: value attributes that will be stored in the log
        :type values: str
        :raises FileNotFoundError: If the log file does not exist
        :raises errors.CsvMismatchedColumnsError: If the number of values
        does not match the number of columns
        """
        self.csv_log_index(log_name, key)
        self.csv_log_values(log_name, *values)

    def save_model(self, model_name, state, model_location=None):
        """
        Save the model.

        :param model_name: name of the file in which the model is saved
        :type model_name: str
        :param state: state of the training including the model state dict, the
            optimizer state dict and the scheduler state dict, the epoch and
            the accuracy
        :type state: dict[str, any]
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
        self, model_name, state, accu, target_accu=0.6, model_location=None
    ):
        """
        Save the model if it's accuracy reaches a given threshold.

        :param model_name: name of the file in which the model is saved
        :type model_name: str
        :param state: state of the training including the model state dict, the
            optimizer state dict and the scheduler state dict, the epoch and
            the accuracy
        :type state: dict[str, any]
        :param accu: test accuracy of the model
        :type accu: float
        :param target_accu: accuracy threshold. Defaults to ``0.6``.
        :type target_accu: float
        :param model_location: location where the model is saved. If it is
            ``None``, then the model will be saved into the ``"checkpoints"``
            folder. Defaults to ``None``.
        :type model_location: pathlib.Path | None
        """
        if accu > target_accu:
            with self.increase_indent_context():
                self.info(f"above {target_accu:.2%}")
            self.save_model(f"{model_name}-{accu:.4f}", state, model_location)

    def save_image(self, image_name, image, image_location=None):
        """
        Save the image.

        :param image_name: name of the output image
        :type image_name: str | pathlib.Path
        :param image: image to save
        :type image: numpy.ndarray
        :param image_location: location where the image is saved. Defaults to ``None``.
        :type image_location: str | pathlib.Path
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
