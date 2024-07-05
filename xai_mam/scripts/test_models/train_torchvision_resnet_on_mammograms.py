"""
Train ResNet50 in mammograms
============================

Train ResNet50 model defined in package torchvision on mammogram dataset.

The code is inspired from [#]_.

References & Footnotes
======================

.. [#] https://www.kaggle.com/code/vortexkol/breast-cancer-resnet-50-bm
"""
import dataclasses as dc
import datetime
import warnings
from time import time

import cv2
import hydra
import numpy as np
import omegaconf
import pandas as pd
from icecream import ic
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import SubsetRandomSampler
from torchinfo import summary

from xai_mam.dataset.dataloaders import CustomDataModule
from xai_mam.scripts.test_models.helpers.log import print_set_information, \
    print_set_overlap
from xai_mam.scripts.test_models.helpers.model import Model
from xai_mam.scripts.test_models.helpers.plot import create_plot_from_data_frame
from xai_mam.scripts.test_models.helpers.split import patient_split_data, \
    random_split_data
from xai_mam.scripts.test_models.helpers.train import train_with_lists
from xai_mam.utils.config.types import DataConfig, DatasetConfig, Gpu
from xai_mam.utils.config.resolvers import resolve_run_location, \
    resolve_override_dirname, resolve_create
from xai_mam.utils.config.script_main import JobConfig
from xai_mam.utils.environment import get_env
from xai_mam.utils.log import ScriptLogger


@dc.dataclass
class Config:
    batch_size: int
    data: DataConfig
    epochs: int
    learning_rate: float
    gpu: Gpu
    n_angles: int
    seed: int
    train_feature_extraction: bool
    train_test: bool
    train_validation: bool
    use_dropouts: bool
    job: JobConfig


def get_samplers(
        data_module: CustomDataModule, train_validation: bool = True, seed: int = 0,
) -> tuple[SubsetRandomSampler, SubsetRandomSampler]:
    if train_validation:
        _, (train_sampler, validation_sampler) = next(iter(data_module.folds))
    else:
        train_indices, validation_indices = train_test_split(
            np.arange(len(data_module.train_data)),
            test_size=0.15,
            random_state=seed,
            shuffle=True,
            stratify=data_module.train_data.targets,
        )
        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)

    return train_sampler, validation_sampler


def read_image(data: DatasetConfig, target: str, n_angles: int) -> dict:
    info = {}
    metadata = pd.read_csv(data.metadata.file, **data.metadata.parameters.to_dict())
    metadata = metadata[~metadata[(target, "label")].isna()]
    # data.image_dir = data.image_dir.parent.parent / "pngs"
    for image_index, image_information in metadata.iterrows():
        suffix = (
            f"-{image_information[('mammogram_properties', 'image_number')]}"
            if target == "benign_vs_malignant"
            else ""
        )
        # suffix = ""
        image_name = f"{image_index[1]}{suffix}"
        image_path = data.image_dir / f"{image_name}{data.image_properties.extension}"

        if image_path.suffix == ".npz":
            img = np.load(image_path)["image"]

            if len(img.shape) == 2:
                # convert 1 channel image to 3 channel image
                img = np.tile(img, (3, 1, 1)).transpose(1, 2, 0)
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        img = cv2.resize(img, (224, 224))  # resize image
        rows, cols, color = img.shape
        info[image_name] = {}
        for angle in range(0, n_angles, 8):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle,
                                        1)
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            info[image_name][angle] = img_rotated
    return info


def read_label(data: DatasetConfig, target: str, n_angles: int) -> dict:
    metadata = pd.read_csv(data.metadata.file, **data.metadata.parameters.to_dict())
    metadata = metadata[~metadata[(target, "label")].isna()]
    info = {}
    for image_index, image_information in metadata.iterrows():
        suffix = (
            f"-{image_information[('mammogram_properties', 'image_number')]}"
            if target == "benign_vs_malignant"
            else ""
        )
        # suffix = ""
        image_name = f"{image_index[1]}{suffix}"
        info[image_name] = {}
        for angle in range(0, n_angles, 8):
            info[image_name][angle] = 1 if image_information[(target, "label")] == "M" else 0
    return info


def run(cfg: Config, logger: ScriptLogger):
    logger.info(ic.format(cfg))
    logger.info(f"Use dataset: {cfg.data.set.name}")

    image_information = read_image(cfg.data.set, cfg.data.set.target.name, cfg.n_angles)
    label_information = read_label(cfg.data.set, cfg.data.set.target.name, cfg.n_angles)

    if not cfg.train_test and cfg.train_validation:
        warnings.warn(
            f"Train-test split is set to random. Therefore, it is meaningless "
            f"to use stratified train-validation. Setting train-validation to false."
        )
        cfg.train_validation = False

    ids = label_information.keys()  # ids = acceptable labeled ids
    X = []
    Y = []
    all_image_ids = []
    for id in ids:
        for angle in range(0, cfg.n_angles, 8):
            X.append(image_information[id][angle].transpose(2, 0, 1))
            Y.append(label_information[id][angle])
            all_image_ids.append(id)
    X = np.array(X)
    Y = np.array(Y)
    all_image_ids = np.array(all_image_ids)

    if cfg.train_test:
        x_train, y_train, x_test, y_test, train_ids, test_ids, train_indices = patient_split_data(
            all_image_ids,
            Y,
            image_information, label_information,
        )
    else:
        x_train, y_train, x_test, y_test, train_ids, test_ids, train_indices = random_split_data(
            all_image_ids, np.arange(len(X)), X, Y,
        )

    if cfg.train_validation:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = patient_split_data(
            train_ids,
            np.array([label_information[id][0] for id in train_ids]),
            image_information, label_information,
        )
    else:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = random_split_data(
            all_image_ids, train_indices, X, Y
        )

    print_set_information(
        logger,
        "all",
        np.unique(all_image_ids),
        np.array(
            [list(label_information[id].values()) for id in label_information]
        ).flatten()
    )
    print_set_information(logger, "train", np.unique(train_ids), y_train)
    print_set_information(logger, "validation", np.unique(validation_ids), y_validation)
    print_set_information(logger, "test", np.unique(test_ids), y_test)

    print_set_overlap(
        logger,
        ("train", train_ids),
        ("validation", validation_ids),
        ("test", test_ids),
    )

    data_module = hydra.utils.instantiate(cfg.data.datamodule)

    # logger.log_data_module(data_module)
    #
    # train_sampler, validation_sampler = get_samplers(
    #     data_module, cfg.train_validation, cfg.seed,
    # )
    #
    # train_loader = data_module.train_dataloader(
    #     batch_size=cfg.batch_size, sampler=train_sampler,
    # )
    # validation_loader = data_module.validation_dataloader(
    #     batch_size=cfg.batch_size, sampler=validation_sampler,
    # )

    train_start = time()
    logger.info("Dataloader information:")
    # logger.log_dataloader(
    #     ("train", train_loader),
    #     ("validation", validation_loader)
    # )

    model = Model(2, 3, cfg.use_dropouts, cfg.train_feature_extraction)
    model.to(cfg.gpu.device_instance)

    logger.info(summary(
        model,
        input_size=(
            3,
            data_module.dataset.image_properties.height,
            data_module.dataset.image_properties.width,
        ),
        col_names=("input_size", "output_size", "kernel_size"),
        depth=5,
        batch_dim=0,
        device=cfg.gpu.device_instance,
        verbose=0,
    ))

    for param_name, param in model.named_parameters():
        logger.debug(f"{param_name}: {param.requires_grad}")

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    metrics = pd.DataFrame(
        [],
        columns=[
            "train_loss", "train_accuracy",
            "validation_loss", "validation_accuracy"
        ],
        index=pd.Index([], name="epoch"),
    )
    metrics_for_plot = pd.DataFrame(columns=["epoch", "set", "loss", "accuracy"])

    for epoch in range(1, cfg.epochs + 1):
        # train_loss, train_accuracy = train_model(
        #     model, train_loader, criterion, optimizer, gpu=cfg.gpu
        # )
        train_loss, train_accuracy = train_with_lists(
            model, x_train, y_train, cfg.batch_size, criterion, cfg.gpu, optimizer,
        )
        train_result = (
            f"Epoch [{epoch:>2} / {cfg.epochs}] "
            f"train loss = {train_loss:.4f} train accu = {train_accuracy:3.2%}"
        )
        metrics_for_plot.loc[2 * epoch - 1] = [epoch, "train", train_loss,
                                               train_accuracy]

        # validation_loss, validation_accuracy = train_model(
        #     model, validation_loader, criterion, gpu=cfg.gpu
        # )
        validation_loss, validation_accuracy = train_with_lists(
            model, x_validation, y_validation, cfg.batch_size, criterion, cfg.gpu,
        )
        logger.info(f"{train_result} valid loss = {validation_loss:.4f} "
                    f"valid accu = {validation_accuracy:3.2%}")
        metrics.loc[epoch] = [
            train_loss, train_accuracy, validation_loss, validation_accuracy
        ]
        metrics_for_plot.loc[2 * epoch] = [
            epoch, "validation", validation_loss, validation_accuracy
        ]

    metrics.to_csv(logger.log_location / "metrics.csv")

    # test_loader = data_module.test_dataloader(batch_size=cfg.batch_size)
    # test_loss, test_accuracy = train_model(model, test_loader, criterion, gpu=cfg.gpu)
    test_loss, test_accuracy = train_with_lists(
        model, x_test, y_test, cfg.batch_size, criterion, cfg.gpu,
    )
    logger.info(f"Test loss = {test_loss} test accuracy = {test_accuracy:.2%}")

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
    metrics.to_csv(logger.log_location / "metrics.csv")

    logger.info(
        f"FINISHED TRAINING in {datetime.timedelta(seconds=int(time() - train_start))}"
    )


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIG_PATH"),
    config_name="script_train_torchvision_resnet_on_mammograms",
)
def main(cfg: Config):
    logger = ScriptLogger(__name__)
    try:
        warnings.showwarning = lambda message, *args: logger.exception(
            message, warn_only=True
        )

        cfg = omegaconf.OmegaConf.to_object(cfg)

        run(cfg, logger)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    # add type validations
    config_store_ = DataConfig.init_store()
    config_store_.store(name="_config_validation", node=Config)

    # add resolvers
    resolve_create()
    resolve_override_dirname()
    resolve_run_location()

    main()
