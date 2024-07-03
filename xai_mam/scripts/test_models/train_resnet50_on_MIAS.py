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
import torch
from hydra.core import hydra_config
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torchinfo import summary
from torchvision.models import resnet50
from tqdm import tqdm

from xai_mam.utils.config.resolvers import resolve_run_location, resolve_create
from xai_mam.utils.environment import get_env
from xai_mam.utils.log import ScriptLogger


@dc.dataclass
class Config:
    epochs: int
    batch_size: int
    dropout: bool
    train_test: bool
    train_validation: bool
    train_feature_extraction: bool
    no_angles: int = 360
    data_path: Path = Path(get_env("DATA_ROOT"), "MIAS")


def read_image(data_path: Path, no_angles: int):
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


def read_label(data_path: Path, no_angles: int):
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
    return (info)


def select_elements(indices, *dicts) -> list:
    results = []
    for current_dict in dicts:
        results.append([])
        for i in indices:
            if type(current_dict[i]) is dict:
                values = current_dict[i].values()
            else:
                values = [current_dict[i]]
            results[-1].extend(list(values))
        results[-1] = np.array(results[-1])
    return results


def patient_split_data(
        patient_ids, labels, *data, test_size=0.15, random_state=2021, shuffle=True
):
    train_indices, test_indices = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=labels
    )
    return (
        *select_elements(train_indices, *data),
        *select_elements(test_indices, *data),
        train_indices,
        test_indices
    )


def random_split_data(
        patient_ids, indices, *data, test_size=0.15, random_state=2021, shuffle=True
):
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return (
        *select_elements(train_indices, *data),
        *select_elements(test_indices, *data),
        patient_ids[train_indices],
        patient_ids[test_indices],
        train_indices,
    )


class Model(nn.Module):
    def __init__(self, dropout: bool = False, train_feature_extraction: bool = False):
        super().__init__()
        self.features = resnet50(
            weights='ResNet50_Weights.IMAGENET1K_V2',
        )
        n_fc_filters = self.features.fc.in_features
        if not train_feature_extraction:
            with torch.no_grad():
                for m in self.features.parameters():
                    m.requires_grad = False
        self.features.fc = nn.Identity()
        if dropout:
            self.classification = nn.Sequential(
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(n_fc_filters, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(256),
                nn.Linear(256, 2),
                nn.Sigmoid(),
            )
        else:
            self.classification = nn.Linear(n_fc_filters, 2)

    def forward(self, x: torch.Tensor):
        out = self.features(x)
        out = self.classification(out)
        return out


def fit(model, X, y, batch_size, criterion, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    running_loss = 0.0
    running_accuracy = 0
    for i in range(0, len(X), batch_size):
        images = torch.tensor(X[i:i + batch_size]).to(torch.float32).to(
            "cuda").permute(0, 3, 1, 2)
        target = torch.LongTensor(y[i:i + batch_size]).to("cuda")

        outputs = model(images)

        # zero the parameter gradients
        if optimizer is not None:
            optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(outputs, target)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        predictions = torch.max(outputs.data, 1)[1]
        running_accuracy += (predictions == target).sum().item()
    loss = running_loss / len(X)
    accu = running_accuracy / len(X)

    return loss, accu


def print_set_information(logger, set_name, patient_ids, labels):
    logger.info(f"{set_name}")
    logger.increase_indent()

    logger.info(f"Number of patients: {len(patient_ids)}")
    logger.info(f"Number of images: {len(labels)}")
    logger.info(f"IDs: {patient_ids}")
    distribution = pd.DataFrame(np.unique(labels, return_counts=True)).T
    distribution.columns = ["label", "count"]
    distribution["perc"] = distribution["count"] / distribution["count"].sum()
    logger.info(
        distribution.to_string(index=False, formatters={"perc": "{:3.2%}".format})
    )

    logger.decrease_indent()


def run(cfg: Config, logger: ScriptLogger):
    result_dir = Path(hydra_config.HydraConfig.get().runtime.output_dir)

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

    if cfg.train_test:
        x_train, y_train, x_test, y_test, train_ids, test_ids = patient_split_data(
            patient_ids,
            [label_info[id][0] for id in patient_ids],
            image_info, label_info,
        )
    else:
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
        x_train, y_train, x_test, y_test, train_ids, test_ids, train_indices = random_split_data(
            all_image_ids, range(len(X)), X, Y,
        )

    logger.info(x_train[0].shape)

    model = Model()
    model.to("cuda")
    summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "kernel_size", "num_params"],
    )

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if cfg.train_test:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids = patient_split_data(
            train_ids,
            [label_info[id][0] for id in train_ids],
            image_info, label_info,
        )
    else:
        x_train, y_train, x_validation, y_validation, train_ids, validation_ids, _ = random_split_data(
            all_image_ids, train_indices, X, Y
        )

    print_set_information(logger, "train", np.unique(train_ids), y_train)
    print_set_information(logger, "validation", np.unique(validation_ids), y_validation)
    print_set_information(logger, "test", np.unique(test_ids), y_test)

    logger.info(f"train ∩ validation: {set(train_ids) & set(validation_ids)}")
    logger.info(f"train ∩ test: {set(train_ids) & set(test_ids)}")
    logger.info(f"test ∩ validation: {set(test_ids) & set(validation_ids)}")

    metrics = {
        "loss.train": [],
        "loss.validation": [],
        "acc.train": [],
        "acc.validation": [],
    }
    for epoch in range(1, cfg.epochs + 1):
        loss, accu = fit(model, x_train, y_train, cfg.batch_size, criterion, optimizer)
        metrics["loss.train"].append(loss)
        metrics["acc.train"].append(accu)
        train_results = (
            f"Epoch [{epoch:3d}/{cfg.epochs:3d}] "
            f"train loss: {loss:.3f} "
            f"train accu: {accu:.3%}"
        )

        loss, accu = fit(model, x_validation, y_validation, cfg.batch_size, criterion, optimizer)
        metrics["loss.validation"].append(loss)
        metrics["acc.validation"].append(accu)
        logger.info(f"{train_results} valid loss: {loss:.3f} valid accu: {accu:.3%}")

    # plot losses
    plt.plot(metrics["loss.train"], label="train")
    plt.plot(metrics["loss.validation"], label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(result_dir / "loss.png", dpi=300)

    # plot accuracies
    plt.clf()
    plt.plot(metrics["acc.train"], label="train")
    plt.plot(metrics["acc.validation"], label="validation")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.savefig(result_dir / "acc.png", dpi=300)
    # save metrics into a file
    pd.DataFrame(
        metrics, index=pd.Index(range(1, cfg.epochs + 1), name="epoch")
    ).to_csv(result_dir / "metrics.csv")

    loss, accu = fit(model, x_test, y_test, cfg.batch_size, criterion)
    logger.info(
        f"Test loss: {loss:.3f} "
        f"test accu: {accu:.3%}"
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
