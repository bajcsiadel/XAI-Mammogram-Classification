"""
Train ResNet50 on MIAS dataset
==============================

Train torchvision ResNet50 on MIAS dataset.
Inspired from code [#1]_.

Using Test time augmentation (TTA) [#2]_ to improve the performance of the model.

References & Footnotes
======================

.. [#1] https://www.kaggle.com/code/vortexkol/breast-cancer-resnet-50-bm
.. [#2] https://doi.org/10.1007/s00521-023-09165-w
"""
import dataclasses as dc
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core import hydra_config
from icecream import ic
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Adam
from torchinfo import summary
from tqdm import tqdm

from xai_mam.scripts.test_models.helpers.log import print_set_information, \
    print_set_overlap
from xai_mam.scripts.test_models.helpers.model import Model
from xai_mam.scripts.test_models.helpers.plot import create_plot_from_data_frame
from xai_mam.scripts.test_models.helpers.split import patient_split_data
from xai_mam.scripts.test_models.helpers.train import train_with_lists
from xai_mam.utils.config.types import Gpu
from xai_mam.utils.config.resolvers import resolve_run_location, resolve_create
from xai_mam.utils.environment import get_env
from xai_mam.utils.log import ScriptLogger


@dc.dataclass
class Config:
    epochs: int
    batch_size: int
    learning_rate: float
    use_dropout: bool
    train_test: bool  # not used
    train_validation: bool  # not used
    train_feature_extraction: bool
    gpu: Gpu
    no_angles: int = 360
    data_path: Path = Path(get_env("DATA_ROOT"), "MIAS")


def read_image(data_path: Path, no_angles: int) -> dict[str, dict[int, np.ndarray]]:
    import cv2
    info = {}
    for i in tqdm(range(322), desc="Reading images"):
        if i < 9:
            image_name = 'mdb00' + str(i + 1)
        elif i < 99:
            image_name = 'mdb0' + str(i + 1)
        else:
            image_name = 'mdb' + str(i + 1)
        image_address = data_path / "pngs" / f"{image_name}.png"
        img = cv2.imread(str(image_address), 1)
        img = cv2.resize(img, (224, 224))  # resize image
        # convert images from [0, 255] to [0, 1]
        img = img.astype(np.float32)
        img /= 255
        rows, cols, color = img.shape
        info[image_name] = {}
        for angle in range(0, no_angles, 8):
            M = cv2.getRotationMatrix2D(
                (cols / 2, rows / 2), angle, 1
            )  # Rotate 0 degree
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            info[image_name][angle] = img_rotated
    return info


def read_label(data_path: Path, no_angles: int) -> dict[str, dict[int, int]]:
    filename = data_path / "data.txt"
    with filename.open("r") as fd:
        text_all = fd.read()
    lines = text_all.split("\n")
    info = {}
    for line in lines:
        words = line.split(' ')
        if len(words) > 3:
            match words[3]:
                case "B":
                    info[words[0]] = {}
                    for angle in range(0, no_angles, 8):
                        info[words[0]][angle] = 0
                case "M":
                    info[words[0]] = {}
                    for angle in range(0, no_angles, 8):
                        info[words[0]][angle] = 1
    return info


def test_model(
    model: nn.Module,
    input_: np.ndarray,
    output_: np.ndarray,
    base_indices: np.ndarray,
    batch_size: int,
    criterion: nn.Module,
    gpu: Gpu,
    logger: ScriptLogger,
) -> tuple[float, float]:
    """
    Train the given model using the provided data loader.

    :param model: model to be trained
    :param input_: input data
    :param output_: output data, must have the same length as base_indices!
    :param base_indices: for each image the index of the base image
    :param batch_size: number of samples per batch
    :param criterion: loss function
    :param gpu: gpu configuration
    :return: tuple containing the average loss and accuracy
    """

    model.eval()

    total_loss = 0.0
    total_predictions = 0.0
    total_samples = 0
    batch_size = 180
    for i in range(0, len(input_), batch_size):
        images = torch.tensor(input_[i:i + batch_size]).to(torch.float32).to(
            gpu.device_instance
        ).permute(0, 3, 1, 2)
        targets = torch.LongTensor(output_[i:i + batch_size]).to(gpu.device_instance)
        aggregate_indices_target = (
            torch
            .LongTensor(base_indices[i:i + batch_size])
            .to(gpu.device_instance)
        )
        aggregate_indices_target -= aggregate_indices_target.min()

        with torch.no_grad():
            outputs = model(images)

        aggregate_indices_output = (
            aggregate_indices_target
            .view(aggregate_indices_target.size(0), 1)
            .expand(-1, outputs.shape[1])
        )

        unique_base_indices = aggregate_indices_output.unique(dim=0)

        logger.debug(f"{outputs.shape = }")
        logger.debug(f"{aggregate_indices_output.shape = }")
        logger.debug(f"{outputs.dtype = }")
        logger.debug(f"{aggregate_indices_output.dtype = }")
        logger.debug(f"{outputs = }")
        logger.debug(f"{aggregate_indices_output = }")

        logger.debug(f"{unique_base_indices.shape = }")

        outputs = (
            torch
            .zeros_like(unique_base_indices, dtype=torch.float)
            .scatter_reduce(0, aggregate_indices_output, outputs, "mean")
        )

        logger.debug(f"{targets.shape = }")
        logger.debug(f"{aggregate_indices_target.shape = }")
        logger.debug(f"{targets.dtype = }")
        logger.debug(f"{aggregate_indices_target.dtype = }")
        logger.debug(f"{targets = }")
        logger.debug(f"{aggregate_indices_target = }")

        logger.debug(f"{unique_base_indices[:, 0].shape = }")

        targets = (
            torch
            .zeros_like(unique_base_indices[:, 0], dtype=targets.dtype)
            .scatter_reduce(0, aggregate_indices_target, targets, "amax")
        )

        logger.debug(f"{outputs.shape = }")
        logger.debug(f"{targets.shape = }")

        loss = criterion(outputs, targets)

        total_predictions += (torch.max(outputs, dim=1)[1] == targets).sum().item()
        total_samples += len(targets)
        total_loss += loss.item()

    return total_loss / total_samples, total_predictions / total_samples


def run(cfg: Config, logger: ScriptLogger):
    result_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(ic.format(cfg))

    logger.info("Read label information")
    label_info = read_label(cfg.data_path, cfg.no_angles)
    logger.info("Read image contents and augment")
    image_info = read_image(cfg.data_path, cfg.no_angles)
    patient_ids = np.unique(list(label_info.keys()))
    labels = np.array([label_info[id][0] for id in patient_ids])

    logger.info(f"Class mapping:\n\tB --> 0\n\tM --> 1")

    print_set_information(
        logger,
        "all",
        patient_ids,
        labels,
    )

    x_train, y_train, x_test, y_test, train_ids, test_ids, train_indices = patient_split_data(
        patient_ids,
        labels,
        image_info, label_info,
    )

    x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = patient_split_data(
        train_ids,
        np.array([label_info[id][0] for id in train_ids]),
        image_info, label_info,
    )

    shuffled_train_indices = np.random.permutation(len(train_indices))
    x_train = x_train[shuffled_train_indices]
    y_train = y_train[shuffled_train_indices]

    validation_base_indices = np.arange(len(validation_ids)).repeat(45)
    test_base_indices = np.arange(len(test_ids)).repeat(45)

    # validation_ids = validation_ids.repeat(45)
    # test_ids = test_ids.repeat(45)
    #
    # # x_train, y_train = shuffle_arrays(x_train, y_train)
    # # x_validation, y_validation, validation_base_indices, validation_ids = shuffle_arrays(x_validation, y_validation, validation_base_indices, validation_ids)
    # # x_test, y_test, test_base_indices, test_ids = shuffle_arrays(x_test, y_test, test_base_indices, test_ids)
    #
    # logger.debug("validation")
    # logger.increase_indent()
    # elems = np.unique(
    #     np.vstack((validation_ids, y_validation, validation_base_indices)).T,
    #     axis=0,
    # )
    # for i in sorted(elems, key=lambda x: int(x[2])):
    #     logger.debug(i)
    # logger.decrease_indent()
    #
    # logger.debug("test")
    # logger.increase_indent()
    # elems = np.unique(
    #     np.vstack((test_ids, y_test, test_base_indices)).T,
    #     axis=0,
    # )
    # for i in sorted(elems, key=lambda x: int(x[2])):
    #     logger.debug(i)
    # logger.decrease_indent()

    logger.info(x_train[0].shape)

    model = Model(2, 3, cfg.use_dropout, cfg.train_feature_extraction)
    model.to(cfg.gpu.device_instance)
    logger.info(summary(
        model,
        input_size=(3, 224, 224),
        col_names=["input_size", "output_size", "kernel_size", "num_params"],
        depth=5,
        batch_dim=0,
        verbose=0,
    ))

    optimizer = Adam(model.features.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print_set_information(logger, "train", np.unique(train_ids), y_train)
    print_set_information(logger, "validation", np.unique(validation_ids), y_validation)
    print_set_information(logger, "test", np.unique(test_ids), y_test)

    print_set_overlap(
        logger,
        ("train", train_ids),
        ("validation", validation_ids),
        ("test", test_ids),
    )

    metrics = {
        "loss.train": [],
        "loss.validation": [],
        "acc.train": [],
        "acc.validation": [],
    }
    metrics_for_plot = pd.DataFrame(columns=["epoch", "set", "loss", "accuracy"])
    for epoch in range(1, cfg.epochs + 1):
        if epoch <= cfg.epochs // 2:
            for m in model.classification.parameters():
                m.requires_grad = False
        else:
            for m in model.features.parameters():
                m.requires_grad = False
            for m in model.classification.parameters():
                m.requires_grad = True
            if epoch == cfg.epochs // 2 + 1:
                optimizer = Adam(model.classification.parameters(), lr=cfg.learning_rate)
        loss, accu = train_with_lists(
            model, x_train, y_train, cfg.batch_size, criterion, cfg.gpu, optimizer,
        )
        metrics["loss.train"].append(loss)
        metrics["acc.train"].append(accu)
        train_results = (
            f"Epoch [{epoch:3d}/{cfg.epochs:3d}] "
            f"train loss: {loss:.4f} "
            f"train accu: {accu:.2%}"
        )
        metrics_for_plot.loc[2 * epoch - 1] = [epoch, "train", loss, accu]

        loss, accu = test_model(
            model,
            x_validation,
            y_validation,
            validation_base_indices,
            cfg.batch_size,
            criterion,
            cfg.gpu,
            logger,
        )
        metrics["loss.validation"].append(loss)
        metrics["acc.validation"].append(accu)
        logger.info(f"{train_results} valid loss: {loss:.4f} valid accu: {accu:.2%}")
        metrics_for_plot.loc[2 * epoch] = [epoch, "validation", loss, accu]

    # plot losses

    create_plot_from_data_frame(
        metrics_for_plot,
        "epoch", "loss",
        "set",
        save_path=logger.log_location / "loss.png",
    )

    create_plot_from_data_frame(
        metrics_for_plot,
        "epoch", "accuracy",
        "set",
        save_path=logger.log_location / "accuracy.png",
    )

    # save metrics into a file
    pd.DataFrame(
        metrics, index=pd.Index(range(1, cfg.epochs + 1), name="epoch")
    ).to_csv(result_dir / "metrics.csv")

    loss, accu = test_model(
        model,
        x_test,
        y_test,
        test_base_indices,
        cfg.batch_size,
        criterion,
        cfg.gpu,
        logger,
    )
    logger.info(
        f"Test loss: {loss:.4f} "
        f"test accu: {accu:.2%}"
    )
    logger.info("FINISHED")


def shuffle_arrays(*arrays):
    shuffled_indices = np.random.permutation(len(arrays[0]))
    return (
        array[shuffled_indices]
        for array in arrays
    )


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIG_PATH"),
    config_name="script_train_resnet50_on_MIAS"
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
