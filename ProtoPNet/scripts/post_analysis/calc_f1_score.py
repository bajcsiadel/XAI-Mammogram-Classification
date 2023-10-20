import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ProtoPNet.config.settings import (
    base_architecture,
    experiment_run,
    img_size,
    test_batch_size,
    test_dir,
    train_batch_size,
    train_dir,
)
from ProtoPNet.util.preprocess import mean, std
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("-gpuid", nargs=1, type=str, default="0")
parser.add_argument("--backbone", action="store_true")
parser.add_argument("-epochs", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid[0]
epochs = args.epochs.split(",")
backbone_only = args.backbone

save_dir = os.path.join(
    "./saved_models/",
    base_architecture,
    experiment_run + ("_backbone" if backbone_only else ""),
)
model_dir = os.path.join(save_dir, "model")
log_csv_path = os.path.join(save_dir, "loss_terms_and_accu.csv")

model_pattern = re.compile(r"^([0-9]+)nopush.*\.pth$")

normalize = transforms.Normalize(mean=mean, std=std)
# datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
)


def evaluate(model, dataloader):
    true_labels = np.array([])
    predicted_labels = np.array([])
    with torch.no_grad():
        for image, label in dataloader:
            input = image.cuda()
            target = label.cuda()

            if backbone_only:
                output = model(input)
            else:
                output, _ = model(input)

            _, predicted = torch.max(output.data, 1)
            true_labels = np.append(true_labels, label.numpy())
            predicted_labels = np.append(
                predicted_labels, predicted.cpu().numpy()
            )

    del input
    del target
    del output
    del predicted

    micro_f1 = f1_score(true_labels, predicted_labels, average="micro")
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    return micro_f1, macro_f1


if __name__ == "__main__":
    print(f"calculating f1 scores on experiment run {experiment_run}")

    log_df = pd.read_csv(log_csv_path)
    log_df["train_micro_f1"] = log_df.get(
        "train_micro_f1", pd.Series(dtype=float)
    )
    log_df["train_macro_f1"] = log_df.get(
        "train_macro_f1", pd.Series(dtype=float)
    )
    log_df["test_macro_f1"] = log_df.get(
        "test_macro_f1", pd.Series(dtype=float)
    )
    log_df["test_micro_f1"] = log_df.get(
        "test_micro_f1", pd.Series(dtype=float)
    )

    for file in os.listdir(model_dir):
        matcher = model_pattern.match(file)
        if matcher:  # and matcher.group(1) in epochs:
            epoch_number = matcher.group(1)
            load_path = os.path.join(model_dir, file)
            model = torch.load(
                load_path, map_location=lambda storage, _loc: storage
            ).cuda()

            print(f"epoch {epoch_number}:")

            print("\ttrain")
            train_micro_f1, train_macro_f1 = evaluate(model, train_loader)
            print(f"\t\tmicro f1: {train_micro_f1}")
            print(f"\t\tmacro f1: {train_macro_f1}")

            print("\ttest")
            test_micro_f1, test_macro_f1 = evaluate(model, test_loader)
            print(f"\t\tmicro f1: {test_micro_f1}")
            print(f"\t\tmacro f1: {test_macro_f1}")

            f1_scores = pd.Series(
                [train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1],
                index=[
                    "train_micro_f1",
                    "train_macro_f1",
                    "test_macro_f1",
                    "test_micro_f1",
                ],
                name=epoch_number,
            )
            log_df.loc[epoch_number] = f1_scores

    log_df.to_csv(log_csv_path)
