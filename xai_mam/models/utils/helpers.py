import re

import torch
from torch.utils import model_zoo

from xai_mam.utils.environment import get_env


def load_model(model_location, construct_model, logger):
    """
    Load a model from a file.

    :param model_location: location of the file in which the trained model is saved.
    :type model_location: pathlib.Path
    :param construct_model: function to construct the model
    :type construct_model: typing.Callable
    :param logger:
    :type logger: xai_mam.utils.log.TrainLogger
    :return: model instance
    :rtype: (xai_mam.models._base_classes.Model, dict)
    """
    if not model_location.exists():
        raise FileNotFoundError(f"Model file not found at {model_location}")
    saved_state = torch.load(model_location)["state_dict"]
    model_initialization_params = saved_state["model_initialization_parameters"]
    model = construct_model(
        logger=logger,
        **model_initialization_params
    )
    model.load_state_dict(saved_state["model"])

    return model, model_initialization_params


def get_state_dict(
    model_url: str,
    n_color_channels: int = 3,
    prefixes: dict[str, str] | None = None,
    include_fc: bool = False,
    n_classes: int | None = None,
) -> dict:
    """
    Load the model state dict from a pretrained model of ImageNet.

    :param model_url: url to the pretrained model
    :param n_color_channels: number of color channels. Defaults to ``3``.
    :param prefixes: prefix of the layer names. Defaults to ``None``.
    :param include_fc: marks the inclusion or exclusion of fully connected layer
    wights. Defaults to ``False``.
    :param n_classes: number of classes to predict. If ``include_fc`` is ``True``
    and ``n_classes`` is different from the number of classes used in the original
    training, then the first most activated neurons will be used. Defaults to ``None``.
    :return: state dict of the pretrained model
    """
    PRETRAINED_MODELS_DIR = get_env("PRETRAINED_MODELS_DIR")
    pretrained_state_dict = model_zoo.load_url(
        model_url, model_dir=PRETRAINED_MODELS_DIR, map_location=torch.device("cpu")
    )
    # remove the weights of the last layer
    # the number of classes could be different from the original model
    if not include_fc:
        pretrained_state_dict.pop("fc.weight")
        pretrained_state_dict.pop("fc.bias")
    elif n_classes is not None:
        n_pretrained_classes = len(pretrained_state_dict["fc.weight"])
        print(n_pretrained_classes)
        if n_classes < n_pretrained_classes:
            pretrained_state_dict["fc.weight"] = pretrained_state_dict["fc.weight"][
                :n_classes
            ]
            pretrained_state_dict["fc.bias"] = pretrained_state_dict["fc.bias"][
                :n_classes
            ]
        elif n_classes > n_pretrained_classes:
            raise ValueError(
                f"Number of classes {n_classes} higher than the number of classes"
                f" used in the pretraining. Solution switch off the `include_fc`."
            )
    if n_color_channels == 1:
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
