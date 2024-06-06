import warnings

import hydra
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from dotenv import load_dotenv
from torchinfo import summary
from tqdm import tqdm
from torchvision import models

from xai_mam.utils.config import script_main as main_cfg
from xai_mam.utils.environment import get_env
from xai_mam.utils.log import TrainLogger


def train(model, train_loader, optimizer, criterion, device, logger):
    """
    Train the model.

    :param model:
    :type model: torch.nn.Module
    :param train_loader:
    :type train_loader: torch.utils.data.DataLoader
    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param criterion:
    :param device:
    :type device: torch.device
    :param logger:
    :type logger: xai_mam.utils.log.TrainLogger
    :return:
    """
    model.train()
    logger.info("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for data in tqdm(train_loader, total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, predictions = torch.max(outputs.data, 1)
        train_running_correct += (predictions == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = train_running_correct / len(train_loader.dataset)
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device, logger):
    """
    Validate the model.

    :param model:
    :type model: torch.nn.Module
    :param test_loader:
    :type test_loader: torch.utils.data.DataLoader
    :param criterion:
    :param device:
    :type device: torch.device
    :param logger:
    :type logger: xai_mam.utils.log.TrainLogger
    :return:
    """
    model.eval()
    logger.info("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, predictions = torch.max(outputs.data, 1)
            valid_running_correct += (predictions == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = valid_running_correct / len(test_loader.dataset)
    return epoch_loss, epoch_acc


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIG_PATH"),
    config_name="script_train_resnet_torchvision_config",
)
def main(cfg: main_cfg.Config):
    with TrainLogger(__name__, cfg.outputs) as logger:
        try:
            warnings.showwarning = lambda message, *args: logger.exception(
                message, warn_only=True
            )

            logger.log_command_line()

            cfg = omegaconf.OmegaConf.to_object(cfg)

            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed(cfg.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)

            run(cfg, logger)
        except Exception as e:
            logger.exception(e)


def run(cfg, logger):
    data_module = hydra.utils.instantiate(cfg.data.datamodule)
    data_module.log_data_information(logger)
    epochs = cfg.model.phases["joint"].epochs
    batch_size = 16
    learning_rate = 0.01
    device = torch.device(
        f"{cfg.gpu.device}" +
        f":{cfg.gpu.device_ids.split(',')[0]}" if cfg.gpu.device == "cuda" else ""
    )

    train_loader = data_module.train_dataloader(batch_size=batch_size)
    validation_loader = data_module.validation_dataloader(batch_size=batch_size)

    if not hasattr(models, cfg.model.network.name):
        raise ValueError(
            f"Model {cfg.model.network.name!r} not implemented in torchvision."
        )

    weights = [
        weight
        for weight in dir(models)
        if weight.lower() == f"{cfg.model.network.name}_weights"
    ]
    if not weights:
        raise ValueError(
            f"Weights for model {cfg.model.network.name!r} not "
            f"implemented in torchvision."
        )
    else:
        weights = getattr(models, weights[0]).DEFAULT
    model = getattr(models, cfg.model.network.name)(weights=weights)
    if cfg.data.set.image_properties.color_channels != 3:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, data_module.dataset.number_of_classes)
    model.to(device)

    logger.info(model)
    exit()
    logger.info(summary(
        model,
        input_size=(
            data_module.dataset.image_properties.color_channels,
            data_module.dataset.image_properties.height,
            data_module.dataset.image_properties.width,
        ),
        depth=5,
        batch_dim=0,
        device=device,
        verbose=0,
    ))

    # Optimizer.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Start the training.
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            logger,
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model,
            validation_loader,
            criterion,
            device,
            logger,
        )
        logger.info(
            f"Training loss: {train_epoch_loss:.3f}, "
            f"training acc: {train_epoch_acc:.3%}"
        )
        logger.info(
            f"Validation loss: {valid_epoch_loss:.3f}, "
            f"validation acc: {valid_epoch_acc:.3%}"
        )
        logger.tensorboard.add_scalar("loss/train", train_epoch_loss, epoch)
        logger.tensorboard.add_scalar("loss/eval", valid_epoch_loss, epoch)
        logger.tensorboard.add_scalars("loss", {
            "loss/train": train_epoch_loss,
            "loss/eval": valid_epoch_loss,
        }, epoch)
        logger.tensorboard.add_scalar("accuracy/train", train_epoch_acc, epoch)
        logger.tensorboard.add_scalar("accuracy/eval", valid_epoch_acc, epoch)
        logger.tensorboard.add_scalars("loss", {
            "accuracy/train": train_epoch_acc,
            "accuracy/eval": valid_epoch_acc,
        }, epoch)
        logger.info("-" * 50)

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    load_dotenv()
    main_cfg.init_config_store()
    main()
