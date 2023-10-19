import torch
from database.config import dataset_config

in_channels = dataset_config["color_channels"]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def preprocess(x, mean, std):
    assert x.size(1) == in_channels
    y = torch.zeros_like(x)
    for i in range(in_channels):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    """
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    """
    return preprocess(x, mean=mean, std=std)


def undo_preprocess(x, mean, std):
    assert x.size(1) == in_channels
    y = torch.zeros_like(x)
    for i in range(in_channels):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def undo_preprocess_input_function(x):
    """
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    """
    return undo_preprocess(x, mean=mean, std=std)
