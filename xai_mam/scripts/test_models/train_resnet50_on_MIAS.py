"""
Train ResNet50 on MIAS dataset
==============================

Train torchvision ResNet50 on MIAS dataset.
Inspired from code [#]_.

References & Footnotes
======================

.. [#] https://www.kaggle.com/code/vortexkol/breast-cancer-resnet-50-bm
"""
import dataclasses as dc
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
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
from xai_mam.scripts.test_models.helpers.split import patient_split_data, \
    random_split_data
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
    train_test: bool
    train_validation: bool
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


def run(cfg: Config, logger: ScriptLogger):
    result_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(ic.format(cfg))

    logger.info("Read label information")
    label_info = read_label(cfg.data_path, cfg.no_angles)
    logger.info("Read image contents and augment")
    image_info = read_image(cfg.data_path, cfg.no_angles)
    patient_ids = np.unique(list(label_info.keys()))

    logger.info(f"Class mapping:\n\tB --> 0\n\tM --> 1")

    print_set_information(
        logger,
        "all",
        patient_ids,
        np.array([list(label_info[id].values()) for id in patient_ids]).flatten()
    )

    X = []
    Y = []
    all_image_ids = []
    for patient_id in patient_ids:
        for angle in range(0, cfg.no_angles, 8):
            X.append(image_info[patient_id][angle])
            Y.append(label_info[patient_id][angle])
            all_image_ids.append(patient_id)
    X = np.array(X)
    Y = np.array(Y)
    all_image_ids = np.array(all_image_ids)

    if cfg.train_test:
        x_train, y_train, x_test, y_test, train_ids, test_ids, train_indices = patient_split_data(
            all_image_ids,
            Y,
            image_info, label_info,
        )
    else:
        x_train, y_train, x_test, y_test, train_ids, test_ids, train_indices = random_split_data(
            all_image_ids, np.arange(len(X)), X, Y,
        )

    if cfg.train_validation:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = patient_split_data(
            train_ids,
            np.array([label_info[id][0] for id in train_ids]),
            image_info, label_info,
        )
    else:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = random_split_data(
            all_image_ids, train_indices, X, Y
        )

    if cfg.train_validation:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = patient_split_data(
            train_ids,
            np.array([label_info[id][0] for id in train_ids]),
            image_info, label_info,
        )
    else:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = random_split_data(
            all_image_ids, train_indices, X, Y
        )

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

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
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

        loss, accu = train_with_lists(
            model, x_validation, y_validation, cfg.batch_size, criterion, cfg.gpu,
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

    loss, accu = train_with_lists(
        model, x_test, y_test, cfg.batch_size, criterion, cfg.gpu,
    )
    logger.info(
        f"Test loss: {loss:.4f} "
        f"test accu: {accu:.2%}"
    )
    logger.info("FINISHED")


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
