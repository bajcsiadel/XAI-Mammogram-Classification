import torch
from torch.utils import model_zoo

from ProtoPNet.utils.environment import get_env


def get_state_dict(model_url, color_channels=3, features_only=True):
    """
    Load the model state dict from a pretrained model of ImageNet.

    :param model_url: url to the pretrained model
    :type model_url: str
    :param color_channels: number of color channels. Defaults to ``3``.
    :type color_channels: int
    :param features_only: marks if the model performs only feature extraction.
        Defaults to ``True``.
    :return: state dict of the pretrained model
    :rtype: dict
    """
    PRETRAINED_MODELS_DIR = get_env("PRETRAINED_MODELS_DIR")
    pretrained_state_dict = model_zoo.load_url(
        model_url, model_dir=PRETRAINED_MODELS_DIR
    )
    if features_only:
        pretrained_state_dict.pop("fc.weight")
        pretrained_state_dict.pop("fc.bias")
    if color_channels == 1:
        conv1_w = pretrained_state_dict.pop("conv1.weight")
        conv1_w = torch.sum(conv1_w, dim=1, keepdim=True)
        pretrained_state_dict["conv1.weight"] = conv1_w
    return pretrained_state_dict

