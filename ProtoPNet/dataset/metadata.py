import abc
import dataclasses as dc
import os
import typing as typ
from pathlib import Path

import albumentations as A
import pipe

from ProtoPNet.util import helpers

DATA_DIR = Path(os.getenv("DATASET_LOCATION"))
assert (
    DATA_DIR is not None
), "Please set the environment variable DATASET_LOCATION in .env file"


@dc.dataclass
class DataFilter(helpers.PartiallyFrozenDataClass, abc.ABC):
    _mutable_attrs = ["scope"]
    field: list[str]
    value: typ.Any
    scope: str = ""

    def __post_init__(self):
        if self.scope == "":
            self.scope = "Filter"
            tmp_field = self.field[:]

            self.scope += "".join(
                tmp_field
                | pipe.map(lambda x: x.split("_"))
                | pipe.chain  # flatten the resulting nested list
                | pipe.map(lambda x: x[0].upper() + x[1:].lower())
            )
            match self.value:
                case float():
                    self.scope += str(self.value).replace(".", "_")
                case list():
                    raise ValueError(
                        "DataFilter: List values are not supported for 'VALUE' field"
                    )
                case _:
                    self.scope += str(self.value)
        self.scope += f"/{self.__class__.__name__}"

    @abc.abstractmethod
    def __call__(self, data):
        raise NotImplementedError

    def __lt__(self, other):
        """
        Compare the current filter to another filter. Needed for sorting
        :param other: other DataFilter object
        :type other: DataFilter
        :return:
        :rtype: bool
        """
        return self.scope < other.scope

    @property
    def field_in_df(self):
        if len(self.field) == 1:
            return self.field[0]
        else:
            return tuple(self.field)


class ExactDataFilter(DataFilter):
    def __call__(self, data):
        """
        Apply the filter to the given data
        :param data:
        :type data: pandas.DataFrame
        :return: the filtered data
        :rtype: pandas.DataFrame
        :raises ValueError: if the given data does not
        contain the field specified in the filter
        """
        if self.field_in_df not in data.columns:
            raise ValueError(
                f"The given data does not contain the field '{self.field_in_df}'"
            )

        return data[data[self.field_in_df] == self.value]
