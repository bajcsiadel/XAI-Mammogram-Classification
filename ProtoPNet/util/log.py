import os
from datetime import datetime
import sys
import traceback
import typing as typ

from ProtoPNet.util import errors
from ProtoPNet.util import helpers


class Log:
    """
    Object for managing the log directory
    :param log_dir: The directory where the log will be stored
    :type log_dir: str
    :param log_file: The name of the log file
    :type log_file: str
    """
    def __init__(self, log_dir, log_file="log.txt"):  # Store log in log_dir
        self.__log_dir = log_dir
        self.__log_file = os.path.join(log_dir, log_file)
        self.__logs = dict()

        # Ensure the directories exist
        if not os.path.isdir(self.log_dir):
            helpers.makedir(self.log_dir)
        if not os.path.isdir(self.metadata_dir):
            helpers.makedir(self.metadata_dir)
        if not os.path.isdir(self.checkpoint_dir):
            helpers.makedir(self.checkpoint_dir)

    @property
    def log_dir(self):
        return self.__log_dir

    @property
    def checkpoint_dir(self):
        return os.path.join(self.__log_dir, "checkpoints")

    @property
    def metadata_dir(self):
        return os.path.join(self.__log_dir, "metadata")

    @property
    def log_file(self):
        return self.__log_file

    def log_message(self, msg="", level="INFO"):
        """
        Write a message to the log file
        :param msg: the message string to be written to the log file
        :param level: the level of the message, e.g. INFO, WARNING, ERROR
        """
        if not os.path.isfile(self.log_file):
            # make log file empty if it already exists
            open(self.log_file, "w").close()
        with open(self.log_file, "a") as f:
            for line in msg.splitlines():
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {level}] {line}\n")

    def log_info(self, msg=""):
        """
        Log a info message
        :param msg:
        :type msg: str
        """
        self.log_message(msg, level="INFO")

    def log_warning(self, msg=""):
        """
        Log a warning message
        :param msg:
        :type msg: str
        """
        self.log_message(msg, level="WARNING")

    def log_error(self, msg=""):
        """
        Log an error message
        :param msg:
        :type msg: str
        """
        self.log_message(msg, level="ERROR")

    def log_exception(self, ex, warn_only=False):
        """
        Log an exception
        :param ex:
        :type ex: Exception
        :param warn_only: Defaults to False
        :type warn_only: bool
        :type ex: Exception
        """
        if warn_only:
            log_fn = self.log_warning
        else:
            log_fn = self.log_error
        log_fn(f"{type(ex).__name__}: {ex}")
        log_fn(traceback.format_exc())

    def log_command_line(self):
        command = " ".join(sys.argv)
        screen_name = "mam-ppn-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(os.path.join(self.metadata_dir, "run-experiment.sh"), mode="w") as fd:
            fd.write("#!/bin/bash\n")
            fd.write("\n")
            fd.write(f"cd {os.getenv('PROJECT_ROOT')}\n")
            fd.write("\n")
            fd.write("# check if environment exists\n")
            fd.write("poetry env list > /dev/null\n")
            fd.write("if { [ $? -ne 0 ] && [ -f \"pyproject.toml\" ]; }\n")
            fd.write("then\n")
            fd.write("\tpoetry install\n")
            fd.write("else\n")
            fd.write("\techo \"No pyproject.toml found. Please run this script from the project root.\"\n")
            fd.write("\texit 1\n")
            fd.write("fi\n")
            fd.write("\n")
            fd.write(f"screen -dmS {screen_name}\n")
            fd.write(f"screen -S {screen_name} -X stuff \"poetry run python {command}\"\n")
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
                self.log_info(msg)
            case "WARNING":
                self.log_warning(msg)
            case "ERROR":
                self.log_error(msg)
            case _:
                self.log_message(message)

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
        with open(self.log_dir + f"/{log_name}.csv", "w") as fd:
            fd.write(",".join(key_name + value_names) + "\n")

    def csv_log_index(self, log_name, key):
        if log_name not in self.__logs.keys():
            raise FileNotFoundError("Log does not exist!")
        if type(key) is str:
            key = (key,)
        if len(key) != len(self.__logs[log_name][0]):
            raise errors.CsvMismatchedColumnsError("Not all indices are logged!")
        with open(os.path.join(self.log_dir, f"{log_name}.csv"), "a") as f:
            f.write(",".join(str(v) for v in key) + ",")

    def csv_log_values(self, log_name, *values):
        """
        Log values in an existent log file. The key should be specified in advance by calling create_csv_log
        :param log_name: The name of the log file
        :type log_name: str
        :param values: value attributes that will be stored in the log
        :type values: str
        :raises FileNotFoundError: If the log file does not exist
        :raises errors.CsvMismatchedColumnsError: If the number of values does not match the number of columns
        """
        if log_name not in self.__logs.keys():
            raise FileNotFoundError("Log does not exist!")
        if len(values) != len(self.__logs[log_name][1]):
            raise errors.CsvMismatchedColumnsError(f"Not all required values are logged! "
                                                   f"Expected {len(self.__logs[log_name][1])}, got {len(values)}")
        # Write a new line with the given values
        with open(os.path.join(self.log_dir, f"{log_name}.csv"), "a") as f:
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
        :raises errors.CsvMismatchedColumnsError: If the number of values does not match the number of columns
        """
        self.csv_log_index(log_name, key)
        self.csv_log_values(log_name, *values)
