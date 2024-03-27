from .file_errors import (
    AbstractClassError,
    CsvMismatchedColumnsError,
    UnsupportedExtensionError,
)


class MissingEnvironmentVariableError(Exception):
    ...
