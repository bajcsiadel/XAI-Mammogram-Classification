import albumentations as A
import gin
import os

import dataclasses as dc
import typing as typ

from ProtoPNet.util import helpers

DATA_DIR = os.getenv("DATASET_LOCATION")
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
    def __init__(self, TRAIN=None, PUSH=None, TEST=None, DISABLED=False):
        self.__TRAIN = TRAIN or []
        self.__PUSH = PUSH or []
        self.__TEST = TEST or []
        self.DISABLED = DISABLED

    def __get_property(self, name):
        if self.DISABLED:
            return []
        attr_name = list(filter(lambda x: x.endswith(f"__{name}"), self.__dict__.keys()))[0]
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
        return list(map(str, augmentations))

    def to_json(self):
        return {
            "TRAIN": self.__augmentations_to_json(self.__TRAIN),
            "PUSH": self.__augmentations_to_json(self.__PUSH),
            "TEST": self.__augmentations_to_json(self.__TEST),
            "DISABLED": self.DISABLED,
        }


@dc.dataclass(frozen=True)
class _DataVersion:
    NAME: str
    DIR: str
    MEAN: typ.Tuple[float] = (0.0, )
    STD: typ.Tuple[float] = (0.0, )


@dc.dataclass(frozen=True)
class _MetadataInformation:
    FILE: str
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
    ROOT_DIR: str
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
class DataFilter(_PartiallyFrozenDataClass):
    _MUTABLE_ATTRS = ["SCOPE"]
    FIELD: str | tuple[str, ...]
    VALUE: typ.Any
    SCOPE: str = ""

    def __post_init__(self):
        if self.SCOPE == "":
            self.SCOPE = "Filter"
            if type(self.FIELD) is tuple:
                self.SCOPE += "".join(map(lambda x: x[0].upper() + x[1:].lower(), self.FIELD))
            match self.VALUE:
                case float():
                    self.SCOPE += str(self.VALUE).replace(".", "_")
                case list():
                    raise ValueError("DataFilter: List values are not supported for 'VALUE' field")
                case _:
                    self.SCOPE += str(self.VALUE)
        self.SCOPE += "/DataFilter"

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
MIAS_ROOT_DIR = os.path.join(DATA_DIR, "MIAS")
DDSM_ROOT_DIR = os.path.join(DATA_DIR, "DDSM")
DATASETS: typ.Dict[str, DatasetInformation] = {
    "MIAS": DatasetInformation(
        NAME="MIAS",
        ROOT_DIR=MIAS_ROOT_DIR,
        VERSIONS={
            "full": {
                "original": _DataVersion(
                    NAME="original",
                    DIR=os.path.join(MIAS_ROOT_DIR, "pngs"),
                    MEAN=(0.2192, ),
                    STD=(0.2930, ),
                ),
                "masked": _DataVersion(
                    NAME="masked",
                    DIR=os.path.join(MIAS_ROOT_DIR, "masked_images", "original"),
                    MEAN=(0.1651, ),
                    STD=(0.2741, ),
                ),
                "masked_preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=os.path.join(MIAS_ROOT_DIR, "masked_images", "clahe"),
                    MEAN=(0.1497, ),
                    STD=(0.2639, ),
                ),
            },
            "cropped": {
                "original": _DataVersion(
                    NAME="original",
                    DIR=os.path.join(MIAS_ROOT_DIR, "pngs"),
                    MEAN=(0.2192, ),
                    STD=(0.2930, ),
                ),
                "masked": _DataVersion(
                    NAME="masked",
                    DIR=os.path.join(MIAS_ROOT_DIR, "masked_images", "original"),
                    MEAN=(0.1651, ),
                    STD=(0.2741, ),
                ),
                "masked_preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=os.path.join(MIAS_ROOT_DIR, "masked_images", "clahe"),
                    MEAN=(0.1497, ),
                    STD=(0.2639, ),
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
            FILE=os.path.join(MIAS_ROOT_DIR, "extended_data.csv"),
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
                    DIR=os.path.join(DDSM_ROOT_DIR, "images"),
                    MEAN=(0.2192,),
                    STD=(0.2930,),
                ),
                "masked": _DataVersion(
                    NAME="masked",
                    DIR=os.path.join(DDSM_ROOT_DIR, "masked_images", "original"),
                    MEAN=(0.1651,),
                    STD=(0.2741,),
                ),
                "masked_preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=os.path.join(DDSM_ROOT_DIR, "masked_images", "clahe"),
                    MEAN=(0.1497,),
                    STD=(0.2639,),
                ),
            },
            "cropped": {
                "original": _DataVersion(
                    NAME="original",
                    DIR=os.path.join(DDSM_ROOT_DIR, "patches", "original"),
                    MEAN=(0.2192,),
                    STD=(0.2930,),
                ),
                "preprocessed": _DataVersion(
                    NAME="masked_preprocessed",
                    DIR=os.path.join(DDSM_ROOT_DIR, "patches", "hist_eq"),
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
            FILE=os.path.join(DDSM_ROOT_DIR, "extended_data.csv"),
            # PARAMETERS={}
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
