import dataclasses as dc
import inspect
import json
import subprocess
import typing as typ
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import albumentations as A
import numpy as np


def get_current_commit_hash():
    """
    Get the hash of the current commit
    :return:
    :rtype: str
    """
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    return process.communicate()[0].strip()


def get_function_name():
    """
    Get the name of the function that called this function. The first index (1) is the
    function that called this and the second index (3) is the name of that function
    :return:
    :rtype: str
    """
    return inspect.stack()[1][3]


def set_used_images(dataset_config, used_images, target):
    """
    Set the used images for the given dataset config and target
    :param dataset_config:
    :type dataset_config: xai_mam.dataset.metadata.DatasetInformation
    :param used_images:
    :type used_images: str
    :param target:
    :type target: str
    """
    assert (
        target in dataset_config.TARGET_TO_VERSION.keys()
    ), "Target does not exist in dataset!"
    versions_key = dataset_config.TARGET_TO_VERSION[target]

    assert (
        used_images in dataset_config.VERSIONS[versions_key].keys()
    ), f"Used images does not exist in dataset for target {target}!"
    dataset_config.USED_IMAGES = dataset_config.VERSIONS[versions_key][used_images]


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_json"):
            return o.to_json()
        elif dc.is_dataclass(o):
            return dc.asdict(o)
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)


class PartiallyFrozenDataClass:
    _mutable_attrs: typ.ClassVar[typ.List[str]] = []

    def __setattr__(self, key, value):
        # freeze all attributes except the ones in _mutable_attrs
        if key in self._mutable_attrs or key not in self.__dict__:
            super().__setattr__(key, value)
        else:
            raise dc.FrozenInstanceError(f"cannot assign to field {key!r}")


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

    @staticmethod
    def new(data):
        if not isinstance(data, dict):
            data = dict(data)

        converted = DotDict(data)

        for key, value in converted.items():
            if isinstance(value, dict):
                converted[key] = DotDict.new(value)
            elif isinstance(value, list):
                converted[key] = [
                    DotDict.new(item) if isinstance(item, dict) else item
                    for item in value
                ]

        return converted

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RepeatedAugmentation(A.Compose):
    """
    Apply the given transformations n times to the input.

    :param transforms: list of transformations to apply
    :type transforms: list[albumentations.BasicTransform]|albumentations.BaseCompose
    :param p: probability of applying the transformation. Defaults to ``1.0``.
    :type p: float
    :param n: number of times to apply the transformation. Defaults to ``1``.
    :type n: int
    """

    def __init__(self, transforms, p=1.0, n=1):
        super().__init__(transforms, p=p)
        self._n_repeat = n

    @property
    def n_repeat(self):
        return self._n_repeat

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __repr__(self):
        return (
            f"RepeatedAugmentation("
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n={self._n_repeat})"
        )


class Shear(A.Affine):
    """
    A custom augmentation that only applies shear in one direction.

    :param limit: maximum shear to apply (in degrees). Defaults to ``45``.
    :type limit: int
    :param always_apply: whether to always apply the transformation.
    Defaults to ``False``.
    :type always_apply: bool
    :param crop_border: whether to crop the border of the image after
    applying the transformation. Defaults to ``False``.
    :type crop_border: bool
    :param p: probability of applying the transformation. Defaults to ``0.5``.
    :type p: float
    """

    def __init__(self, limit=45, always_apply=False, crop_border=False, p=0.5):
        super().__init__(shear=(-limit, limit), always_apply=always_apply, p=p)
        self.original = deepcopy(self.shear)
        self.crop_border = crop_border
        self.direction = None

    def apply(
        self,
        img,
        matrix=None,
        output_shape: Sequence[int] = (),
        **params: Any,
    ) -> np.ndarray:
        """
        Apply the transform to the image.

        :param img: image matrix
        :type img: numpy.ndarray
        :param matrix: affine transform matrix. Defaults to ``None``.
        :param output_shape: shape of the output. Defaults to ``()``.
        :param params:
        :return: the transformed image
        :rtype: numpy.ndarray
        """
        img_out = super().apply(img, matrix, output_shape, **params)

        if self.crop_border:
            height, width = img.shape[:2]
            corners = np.array(
                [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]
            )
            corners = matrix(corners)

            row_positions = (corners[:, 1] >= 0) * (corners[:, 1] < height)
            col_positions = (corners[:, 0] >= 0) * (corners[:, 0] < width)

            row_values = np.rint(sorted(corners[row_positions, 1])).astype(int)
            col_values = np.rint(sorted(corners[col_positions, 0])).astype(int)

            if self.direction == "x":
                row_values = np.array([0, height])
            else:
                col_values = np.array([0, width])

            if len(row_values) != 2 or len(col_values) != 2:
                return img_out
            img_out = img_out[
                row_values[0]:row_values[1], col_values[0]:col_values[1]
            ]
        return img_out

    def __call__(self, *args, **kwargs):
        index = int(np.rint(np.random.uniform(0, 1)))
        # limiting to shear only in one direction
        self.shear[list(self.shear.keys())[index]] = (0.0, 0.0)
        self.direction = list(self.shear.keys())[(index + 1) % 2]
        result = super().__call__(*args, **kwargs)
        self.shear = deepcopy(self.original)
        return result
