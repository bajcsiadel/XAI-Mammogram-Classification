import logging
import os
import sys
import traceback
import typing as typ
from datetime import datetime
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from icecream import ic

from ProtoPNet.util import errors


class Log(logging.Logger):
    """
    Object for managing the log directory

    :param name: The name of the logger
    :type name: str
    """

    def __init__(self, name):
        super().__init__(name)
        self.parent = logging.root

        self.__log_dir = Path(HydraConfig.get().runtime.output_dir)
        self.__logs = dict()

        # Ensure the directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logging.getLogger().manager.loggerDict[name] = self

    @property
    def log_dir(self):
        return self.__log_dir

    @property
    def checkpoint_dir(self):
        return self.__log_dir / "checkpoints"

    @property
    def metadata_dir(self):
        return self.__log_dir / "metadata"

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
        indent = msg[: len(msg) - len(msg.lstrip())]
        msg = msg.strip()
        for line in msg.splitlines():
            super()._log(
                level,
                f"{indent}{line}",
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

    def log_command_line(self):
        """
        Generate a script that can be used to run the experiment again
        """
        python_file = sys.argv[0]
        params = " ".join(HydraConfig.get().overrides.task)
        screen_name = "mam-ppn-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        bash_script_file = self.metadata_dir / "run-experiment.sh"
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
                f'screen -S {screen_name} -X stuff '
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

    def create_csv_log(self, log_name, key_name, *value_names):
        """
        Create a csv for logging information

        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :type log_name: str
        :param key_name: The name of the attribute that is used as key
            (e.g. epoch number)
        :type key_name: str|typ.Iterable[str]
        :param value_names: The names of the attributes that are logged
        :type value_names: str
        """
        if log_name in self.__logs.keys():
            raise Exception("Log already exists!")
        if type(key_name) is str:
            key_name = (key_name,)
        # Add to existing logs
        self.__logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        (self.log_dir / f"{log_name}.csv").write_text(
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
        with (self.log_dir / f"{log_name}.csv").open(mode="a") as f:
            f.write(",".join(str(v) for v in key) + ",")

    def csv_log_values(self, log_name, *values):
        """
        Log values in an existent log file. The key should be
        specified in advance by calling create_csv_log

        :param log_name: The name of the log file
        :type log_name: str
        :param values: value attributes that will be stored in the log
        :type values: str
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
        with (self.log_dir / f"{log_name}.csv").open(mode="a") as f:
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
