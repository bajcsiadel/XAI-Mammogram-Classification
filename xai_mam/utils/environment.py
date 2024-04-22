import os

from dotenv import load_dotenv

from xai_mam.utils.errors import MissingEnvironmentVariableError


def get_env(env_var_name, must_exist=True):
    """
    Get environment variable value.

    :param env_var_name: name of the environment variable
    :type env_var_name: str
    :param must_exist: if ``True`` an error will be raised if the
            environment variable is not set
    :type must_exist: bool

    :returns: value of the environment variable

    :raises utils.errors.MissingEnvironmentVariableError: if the
            environment variable is not set
    """
    load_dotenv()
    if (env_value := os.getenv(env_var_name)) is not None:
        return env_value

    if must_exist:
        raise MissingEnvironmentVariableError(f"{env_var_name!r} is missing from .env")
