import os
from datetime import datetime
import traceback

from ProtoPNet.util import errors
from ProtoPNet.util import helpers


class Log:
    """
    Object for managing the log directory
    """

    def __init__(self, log_dir: str):  # Store log in log_dir
        self.__log_dir = log_dir
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
        return os.path.join(self.__log_dir, "log.txt")

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

    def log_exception(self, ex):
        """
        Log an exception
        :param ex:
        :type ex: Exception
        """
        self.log_error(f"{type(ex).__name__}: {ex}")
        self.log_error(traceback.format_exc())

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

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key
            (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self.__logs.keys():
            raise Exception("Log already exists!")
        # Add to existing logs
        self.__logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self.log_dir + f"/{log_name}.csv", "w") as fd:
            fd.write(",".join((key_name,) + value_names) + "\n")

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        :raises FileNotFoundError: If the log file does not exist
        :raises errors.CsvMismatchedColumnsError: If the number of values does not match the number of columns
        """
        if log_name not in self.__logs.keys():
            raise FileNotFoundError("Log not existent!")
        if len(values) != len(self.__logs[log_name][1]):
            raise errors.CsvMismatchedColumnsError("Not all required values are logged!")
        # Write a new line with the given values
        with open(os.path.join(self.log_dir, f"{log_name}.csv"), "a") as f:
            f.write(",".join(str(v) for v in (key,) + values) + "\n")
