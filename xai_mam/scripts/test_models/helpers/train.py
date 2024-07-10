import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from xai_mam.utils.config._general_types import Gpu


def train_with_dataloader(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    gpu: Gpu,
    optimizer: Optimizer = None,
) -> tuple[float, float]:
    """
    Train the given model using the provided data loader.

    :param model: model to be trained
    :param data_loader: data loader containing the training data
    :param criterion: loss function
    :param gpu: gpu configuration
    :param optimizer: optimizer to be used during training. Defaults to ``None``
        (in case of validation and testing).
    :return: tuple containing the average loss and accuracy
    """
    mode = "train" if optimizer is not None else "eval"

    if mode == "train":
        model.train()
        grad = torch.enable_grad
    else:
        model.eval()
        grad = torch.no_grad

    total_loss = 0.0
    total_predictions = 0.0
    total_samples = 0
    for images, targets in data_loader:
        images = images.to(gpu.device_instance)
        targets = targets.to(gpu.device_instance)

        with grad():
            outputs = model(images)

        loss = criterion(outputs, targets)

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_predictions += (torch.max(outputs, dim=1)[1] == targets).sum().item()
        total_samples += len(targets)
        total_loss += loss.item()

    return total_loss / total_samples, total_predictions / total_samples


def train_with_lists(
    model: nn.Module,
    input_: np.ndarray,
    output_: np.ndarray,
    batch_size: int,
    criterion: nn.Module,
    gpu: Gpu,
    optimizer: Optimizer = None,
) -> tuple[float, float]:
    """
    Train the given model using the provided input and output data.

    :param model: model to be trained
    :param input_: input data
    :param output_: output data
    :param batch_size: number of samples per batch
    :param criterion: loss function
    :param gpu: gpu configuration
    :param optimizer: optimizer to be used during training. Defaults to ``None``
        (in case of validation and testing).
    :return: tuple containing the average loss and accuracy
    """
    input_tensor = torch.tensor(input_).to(torch.float32).to(
        gpu.device_instance
    ).permute(0, 3, 1, 2)
    output_tensor = torch.LongTensor(output_).to(gpu.device_instance)
    data_loader = DataLoader(
        torch.utils.data.TensorDataset(input_tensor, output_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    results = train_with_dataloader(
        model, data_loader, criterion, gpu, optimizer
    )

    # delete tensors to free the used memory
    del input_tensor, output_tensor, data_loader
    torch.cuda.empty_cache()

    return results
