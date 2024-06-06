import re

import torch
from torch.utils import model_zoo

from xai_mam.utils.environment import get_env


def get_state_dict(model_url, color_channels=3, prefixes=None):
    """
    Load the model state dict from a pretrained model of ImageNet.

    :param model_url: url to the pretrained model
    :type model_url: str
    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param prefixes: prefix of the layer names.
    :type prefixes: dict[str, str] | None
    :return: state dict of the pretrained model
    :rtype: dict
    """
    PRETRAINED_MODELS_DIR = get_env("PRETRAINED_MODELS_DIR")
    pretrained_state_dict = model_zoo.load_url(
        model_url, model_dir=PRETRAINED_MODELS_DIR
    )
    # remove the weights of the last layer
    # the number of classes could be different from the original model
    pretrained_state_dict.pop("fc.weight")
    pretrained_state_dict.pop("fc.bias")
    if color_channels == 1:
        conv1_w = pretrained_state_dict.pop("conv1.weight")
        conv1_w = torch.sum(conv1_w, dim=1, keepdim=True)
        pretrained_state_dict["conv1.weight"] = conv1_w

    if prefixes:
        layer_names = list(pretrained_state_dict.keys())
        for layer_pattern, prefix in prefixes.items():
            if prefix and not prefix.endswith("."):
                prefix = f"{prefix}."
                matching_layers = [
                    layer_name
                    for layer_name in layer_names
                    if re.match(layer_pattern, layer_name)
                ]
                for layer_name in matching_layers:
                    value = pretrained_state_dict.pop(layer_name)
                    pretrained_state_dict[f"{prefix}{layer_name}"] = value

    return pretrained_state_dict
