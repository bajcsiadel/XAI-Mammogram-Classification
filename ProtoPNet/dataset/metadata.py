import abc
import gin
import os
import pipe
from pathlib import Path

import albumentations as A
import dataclasses as dc
import typing as typ

from ProtoPNet.util import helpers

DATA_DIR = Path(os.getenv("DATASET_LOCATION"))
assert DATA_DIR is not None, "Please set the environment variable DATASET_LOCATION in .env file"


class _PartiallyFrozenDataClass:
    _MUTABLE_ATTRS: typ.ClassVar[typ.List[str]] = []

    def __setattr__(self, key, value):
        # freeze all attributes except the ones in _MUTABLE_ATTRS
        if key in self._MUTABLE_ATTRS or key not in self.__dict__:
            super().__setattr__(key, value)
        else:
            raise dc.FrozenInstanceError(f"cannot assign to field '{key}'")


class _Augmentations:
    """
    Class representing the necessary augmentations for a dataset
    :param TRAIN: augmentations applied to the train set. Defaults to None.
    :type TRAIN: typ.Iterable[A.BasicTransform] | None
    :param PUSH: augmentations applied to the push set. Defaults to None
    :type PUSH: typ.Iterable[A.BasicTransform] | None
    :param TEST: augmentations applied to the test set. Defaults to None.
    :type TEST: typ.Iterable[A.BasicTransform]
    :param DISABLED: flag marking if augmentations are turned off. Defaults to False.
    :type DISABLED: bool
    """
    def __init__(self, TRAIN=None, PUSH=None, TEST=None, DISABLED=False):
        self.__TRAIN = TRAIN or []
        self.__PUSH = PUSH or []
        self.__TEST = TEST or []
        self.DISABLED = DISABLED

    def __get_property(self, name):
        if self.DISABLED:
            return []
        attr_name = list(
            self.__dir__()
            | pipe.where(lambda x: x.endswith(f"__{name}")))[0]
        return getattr(self, attr_name)

    def __set_property(self, name):
        raise dc.FrozenInstanceError(f"cannot assign to field '{name}'")

    @property
    def TRAIN(self):
        return self.__get_property(helpers.get_function_name())

    @TRAIN.setter
    def TRAIN(self, _):
        self.__set_property(helpers.get_function_name())

    @property
    def PUSH(self):
        return self.__get_property(helpers.get_function_name())

    @PUSH.setter
    def PUSH(self, _):
        self.__set_property(helpers.get_function_name())

    @property
    def TEST(self):
        return self.__get_property(helpers.get_function_name())

    @TEST.setter
    def TEST(self, _):
        self.__set_property(helpers.get_function_name())

    @staticmethod
    def __augmentations_to_json(augmentations):
        """
        Convert the list of augmentations to a list
        of strings representing each augmentation
        :param augmentations:
        :type augmentations: typ.Iterable[A.BasicTransform]
        :return:
        """
        return list(augmentations | pipe.map(str))

    def to_json(self):
        """
        Get JSON representation of the current object
        :return:
        :rtype: str
        """
        return {
            "TRAIN": self.__augmentations_to_json(self.__TRAIN),
            "PUSH": self.__augmentations_to_json(self.__PUSH),
            "TEST": self.__augmentations_to_json(self.__TEST),
            "DISABLED": self.DISABLED,
        }

    def to_string(self):
        """
        Get a string representing the current object
        :return:
        :rtype: str
        """
        return repr(self)

    def __repr__(self):
        """
        Get a string representing the current object
        :return:
        :rtype: str
        """
        return (f"{self.__class__.__name__}(\n"
                f"\tTRAIN={self.__TRAIN},\n"
                f"\tPUSH={self.__PUSH},\n"
                f"\tTEST={self.__TEST},\n"
                f"\tDISABLED={self.DISABLED},\n"
                f")")
Test = _Augmentations


@dc.dataclass(frozen=True)
class _DataVersion:
    NAME: str
    DIR: Path
    MEAN: typ.Tuple[float] = (0.0,)
    STD: typ.Tuple[float] = (0.0,)


@dc.dataclass(frozen=True)
class _MetadataInformation:
    FILE: Path
    PARAMETERS: typ.Dict[str, typ.Any] = dc.field(default_factory=dict)  # set default to empty dict


@dc.dataclass(frozen=True)
class _ImageInformation:
    EXTENSION: str
    SHAPE: typ.Tuple[int, int]
    COLOR_CHANNELS: int
    MAX_VALUE: int = 255
    AUGMENTATIONS: _Augmentations = dc.field(default_factory=_Augmentations)


@dc.dataclass
class DatasetInformation(_PartiallyFrozenDataClass):
    _MUTABLE_ATTRS: typ.ClassVar[typ.List[str]] = ["USED_IMAGES"]
    ROOT_DIR: Path
    VERSIONS: typ.Dict[str, typ.Dict[str, _DataVersion]]
    METADATA: _MetadataInformation
    IMAGE_PROPERTIES: _ImageInformation
    NAME: str = ""
    TARGET_TO_VERSION: typ.Dict[str, str] = dc.field(default_factory=lambda: {
        "normal_vs_abnormal": "full",
        "benign_vs_malignant": "cropped",
        "normal_vs_benign_vs_malignant": "full",
        # todo: add "normal_vs_benign_vs_malignant" for the cropped version
    })
    USED_IMAGES: _DataVersion | None = None


@gin.configurable
@dc.dataclass
class _DataFilter(_PartiallyFrozenDataClass, abc.ABC):
    _MUTABLE_ATTRS = ["SCOPE"]
    FIELD: str | tuple[str, ...]
    VALUE: typ.Any
    SCOPE: str = ""

    def __post_init__(self):
        if self.SCOPE == "":
            self.SCOPE = "Filter"
            tmp_field = self.FIELD
            if type(tmp_field) is str:
                tmp_field = (tmp_field, )

            self.SCOPE += "".join(
                list(tmp_field)
                | pipe.map(lambda x: x.split("_"))
                | pipe.chain  # flatten the resulting nested list
                | pipe.map(lambda x: x[0].upper() + x[1:].lower())
            )
            match self.VALUE:
                case float():
                    self.SCOPE += str(self.VALUE).replace(".", "_")
                case list():
                    raise ValueError("DataFilter: List values are not supported for 'VALUE' field")
                case _:
                    self.SCOPE += str(self.VALUE)
        self.SCOPE += f"/{self.__class__.__name__}"

    def get_short_field(self):
        """
        Get a short version of the field name
        :return:
        :rtype: str
        """
        match self.FIELD:
            case str():
                return self.FIELD
            case tuple() | list():
                return self.FIELD[-1]
            case _:
                raise TypeError(f"Unexpected type for field: {type(self.FIELD)}")

    @abc.abstractmethod
    def __call__(self, data):
        raise NotImplementedError

    def __lt__(self, other):
        """
        Compare the current filter to another filter. Needed for sorting
        :param other: other DataFilter object
        :type other: _DataFilter
        :return:
        :rtype: bool
        """
        return self.SCOPE < other.SCOPE


class ExactDataFilter(_DataFilter):
    def __call__(self, data):
        """
        Apply the filter to the given data
        :param data:
        :type data: pandas.DataFrame
        :return: the filtered data
        :rtype: pandas.DataFrame
        :raises ValueError: if the given data does not contain the field specified in the filter
        """
        if self.FIELD not in data.columns:
            raise ValueError(f"The given data does not contain the field '{self.FIELD}'")

        return data[data[self.FIELD] == self.VALUE]


# dataset configs
MIAS_ROOT_DIR = DATA_DIR / "MIAS"
DDSM_ROOT_DIR = DATA_DIR / "DDSM"
DATASETS: typ.Dict[str, DatasetInformation] = {
    "MIAS": DatasetInformation(
        NAME="MIAS",
        ROOT_DIR=MIAS_ROOT_DIR,
        VERSIONS={
            "full": {
                "original": _DataVersion(
                    NAME="original",
                    DIR=MIAS_ROOT_DIR / "pngs",
                    MEAN=(0.2192,),
                    STD=(0.2930,),
                ),
                "masked": _DataVersion(
                    NAME="masked",
                    DIR=MIAS_ROOT_DIR / "masked_images" / "original",
                    MEAN=(0.1651,),
                    STD=(0.2741,),
                ),
                "masked_preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=MIAS_ROOT_DIR / "masked_images" / "clahe",
                    MEAN=(0.1497,),
                    STD=(0.2639,),
                ),
            },
            "cropped": {
                "original": _DataVersion(
                    NAME="original",
                    DIR=MIAS_ROOT_DIR / "pngs",
                    MEAN=(0.2192,),
                    STD=(0.2930,),
                ),
                "masked": _DataVersion(
                    NAME="masked",
                    DIR=MIAS_ROOT_DIR / "masked_images" / "original",
                    MEAN=(0.1651,),
                    STD=(0.2741,),
                ),
                "masked_preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=MIAS_ROOT_DIR / "masked_images" / "clahe",
                    MEAN=(0.1497,),
                    STD=(0.2639,),
                ),
            },
        },
        # TARGET_TO_VERSION={
        #     "normal_vs_abnormal": "full",
        #     "benign_vs_malignant": "cropped",
        #     "normal_vs_benign_vs_malignant": "full",
        #     # todo: add "normal_vs_benign_vs_malignant" for the cropped version
        # },
        METADATA=_MetadataInformation(
            FILE=MIAS_ROOT_DIR / "extended_data.csv",
            PARAMETERS={
                "header": [0, 1],
                "index_col": [0, 1],
            },
        ),
        IMAGE_PROPERTIES=_ImageInformation(
            EXTENSION=".npz",
            SHAPE=(1024, 1024),
            COLOR_CHANNELS=1,
            # MAX_VALUE=255,
            AUGMENTATIONS=_Augmentations(
                TRAIN=[],
                PUSH=[],
                TEST=[],
                DISABLED=False,
            ),
        ),
    ),
    "DDSM": DatasetInformation(
        NAME="DDSM",
        ROOT_DIR=DDSM_ROOT_DIR,
        VERSIONS={
            "full": {
                "original": _DataVersion(
                    NAME="original",
                    DIR=DDSM_ROOT_DIR / "images",
                    MEAN=(0.2037,),
                    STD=(0.2284,),
                ),
                "masked": _DataVersion(
                    NAME="masked",
                    DIR=DDSM_ROOT_DIR / "masked_images" / "original",
                    MEAN=(0.1651,),
                    STD=(0.2741,),
                ),
                "masked_preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=DDSM_ROOT_DIR / "masked_images" / "clahe",
                    MEAN=(0.1497,),
                    STD=(0.2639,),
                ),
            },
            "cropped": {
                "original": _DataVersion(
                    NAME="original",
                    DIR=DDSM_ROOT_DIR / "patches" / "original",
                    MEAN=(0.2192,),
                    STD=(0.2930,),
                ),
                "preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=DDSM_ROOT_DIR / "patches" / "hist_eq",
                    MEAN=(0.1497,),
                    STD=(0.2639,),
                ),
            },
        },
        # TARGET_TO_VERSION={
        #     "normal_vs_abnormal": "full",
        #     "benign_vs_malignant": "cropped",
        #     "normal_vs_benign_vs_malignant": "full",
        #     # todo: add "normal_vs_benign_vs_malignant" for the cropped version
        # },
        METADATA=_MetadataInformation(
            FILE=DDSM_ROOT_DIR / "extended_data.csv",
            PARAMETERS={
                "header": [0, 1],
                "index_col": [0, 1],
            },
        ),
        IMAGE_PROPERTIES=_ImageInformation(
            EXTENSION=".png",
            SHAPE=(1024, 1024),
            COLOR_CHANNELS=1,
            # MAX_VALUE=255,
            # AUGMENTATIONS=_Augmentations(
            #     TRAIN=[],
            #     PUSH=[],
            #     TEST=[],
            #     DISABLED=False,
            # ),
        ),
    ),
}
