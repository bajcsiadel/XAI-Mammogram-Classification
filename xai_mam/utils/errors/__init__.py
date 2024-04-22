from xai_mam.utils.errors.file_errors import (
    AbstractClassError,
    CsvMismatchedColumnsError,
    UnsupportedExtensionError,
)


class MissingEnvironmentVariableError(Exception):
    ...
