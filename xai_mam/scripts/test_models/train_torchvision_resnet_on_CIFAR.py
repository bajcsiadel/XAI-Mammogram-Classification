import dataclasses as dc
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import models
from xai_mam.utils.config.resolvers import resolve_create, resolve_run_location

from xai_mam.utils.log import ScriptLogger

from xai_mam.utils.environment import get_env


@dc.dataclass
class Config:
    model_name: str
    epochs: int
    learning_rate: float
    batch_size: int


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer = None,
):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    running_loss = 0.0
    running_accuracy = 0
    elements = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if optimizer is not None:
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predictions = torch.max(outputs.data, 1)
        running_accuracy += (predictions == labels).sum().item()
        elements += len(predictions)
    return running_accuracy / elements, running_loss / elements


def run(cfg: Config, logger: ScriptLogger):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    data_location = Path(get_env("DATA_ROOT"), "CIFAR-10")

    train_set = torchvision.datasets.CIFAR10(root=str(data_location), train=True,
                                             download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )

    test_set = torchvision.datasets.CIFAR10(root=str(data_location), train=False,
                                            download=True, transform=transform)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2
    )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    logger.info(" ".join(f"{classes[labels[j]]:5s}" for j in range(cfg.batch_size)))

    if not hasattr(models, cfg.model_name):
        raise ValueError(
            f"Model {cfg.model_name!r} not implemented in torchvision."
        )

    weights = [
        weight
        for weight in dir(models)
        if weight.lower() == f"{cfg.model_name}_weights"
    ]
    if not weights:
        raise ValueError(
            f"Weights for model {cfg.model_name!r} not "
            f"implemented in torchvision."
        )
    else:
        weights = getattr(models, weights[0]).DEFAULT
    model = getattr(models, cfg.model_name)(weights=weights)
    model.fc = nn.Linear(
        model.fc.in_features, 10
    )
    model.to("cuda")
    logger.info(summary(
        model,
        input_size=(3, 32, 32),
        depth=5,
        batch_dim=0,
        verbose=0,
    ))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    for epoch in range(1, cfg.epochs + 1):  # loop over the dataset multiple times
        accuracy, loss = train_model(model, train_loader, criterion, optimizer)
        result = (f"Epoch [{epoch + 1} / {cfg.epochs}] "
                  f"train loss: {loss:.4f} train accuracy {accuracy:3.2%}")

        accuracy, loss = train_model(model, test_loader, criterion)
        logger.info(f"{result} validation loss: {loss:.4f} "
                    f"validation accuracy{accuracy:3.2%}")

    logger.info("Finished Training")


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIG_PATH"),
    config_name="script_train_torchvision_resnet_on_CIFAR"
)
def main(cfg: Config):
    logger = ScriptLogger(__name__)

    try:
        cfg = OmegaConf.to_object(cfg)
        run(cfg, logger)
    except Exception as e:
        logger.exception(e)


from xai_mam.utils.config import config_store_
config_store_.store(name="_config_validation", node=Config)
resolve_run_location()
resolve_create()
main()
