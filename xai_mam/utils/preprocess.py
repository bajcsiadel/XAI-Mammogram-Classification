import torch


def preprocess(x, mean, std, number_of_channels=1):
    """
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    """
    assert x.size(1) == number_of_channels
    y = torch.zeros_like(x)
    for i in range(number_of_channels):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def undo_preprocess(x, mean, std, number_of_channels=1):
    """
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    """
    assert x.size(1) == number_of_channels
    y = torch.zeros_like(x)
    for i in range(number_of_channels):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y
