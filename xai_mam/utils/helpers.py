import dataclasses as dc
import inspect
import json
import subprocess
import typing as typ
from pathlib import Path

import numpy as np
import torch
from pipe import Pipe


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


class CustomPipe:
    @staticmethod
    @Pipe
    def to_list(iterable):
        """
        Convert an iterable to a list (using with pipe)

        :param iterable:
        :return:
        :rtype: list
        """
        return list(iterable)

    @staticmethod
    @Pipe
    def to_numpy(iterable):
        """
        Convert an iterable to a list (using with pipe)

        :param iterable:
        :return:
        :rtype: list
        """
        return np.array(list(iterable))


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
            raise dc.FrozenInstanceError(f"cannot assign to field '{key}'")


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
