import os

import dataclasses as dc
import typing as typ

DATA_DIR = os.getenv("DATASET_LOCATION")
assert DATA_DIR is not None, "Please set the environment variable DATASET_LOCATION in .env file"


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
    AUGMENTATIONS: typ.List[typ.Any] = dc.field(default_factory=list)


@dc.dataclass
class DatasetInformation:
    __MUTABLE_ATTRS: typ.ClassVar[typ.List[str]] = ["USED_IMAGES"]
    ROOT_DIR: str
    VERSIONS: typ.Dict[str, _DataVersion]
    METADATA: _MetadataInformation
    IMAGE_PROPERTIES: _ImageInformation
    NAME: str = ""
    USED_IMAGES: _DataVersion | None = None

    def __setattr__(self, key, value):
        # freeze all attributes except IMAGE_DIR
        if key in DatasetInformation.__MUTABLE_ATTRS or key not in self.__dict__:
            super().__setattr__(key, value)
        else:
            raise dc.FrozenInstanceError(f"cannot assign to field '{key}'")


# dataset configs
MIAS_ROOT_DIR = os.path.join(DATA_DIR, "MIAS")
DDSM_ROOT_DIR = os.path.join(DATA_DIR, "DDSM")
DATASETS: typ.Dict[str, DatasetInformation] = {
    "MIAS": DatasetInformation(
        NAME="MIAS",
        ROOT_DIR=MIAS_ROOT_DIR,
        VERSIONS={
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
        ),
    ),
    "DDSM": DatasetInformation(
        NAME="DDSM",
        ROOT_DIR=DDSM_ROOT_DIR,
        VERSIONS={
            "original": _DataVersion(
                NAME="original",
                DIR=os.path.join(DDSM_ROOT_DIR, "images"),
            ),
            "masked": _DataVersion(
                NAME="masked",
                DIR=os.path.join(DDSM_ROOT_DIR, "masked_images", "original"),
            ),
            "masked_preprocessed": _DataVersion(
                NAME="masked_preprocessed",
                DIR=os.path.join(DDSM_ROOT_DIR, "masked_images", "clahe"),
            ),
        },
        METADATA=_MetadataInformation(
            FILE=os.path.join(DDSM_ROOT_DIR, "extended_data.csv"),
            # PARAMETERS={}
        ),
        IMAGE_PROPERTIES=_ImageInformation(
            EXTENSION=".png",
            SHAPE=(1024, 1024),
            COLOR_CHANNELS=1,
            # MAX_VALUE=255,
        ),
    ),
}
