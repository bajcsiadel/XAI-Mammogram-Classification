import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from database.dataloader import test_loader
from ProtoPNet.config.settings import (
    base_architecture,
    classes,
    experiment_run,
)
from ProtoPNet.util.helpers import makedir
from sklearn.metrics import confusion_matrix

experiment_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
model_dir = os.path.join(experiment_dir, "model")

if __name__ == "__main__":
    '''
    Usage: python3 build_confusion_matrix.py -model='10_18nopush0.7822.pth'
    Generates the confusion matrix for classification
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpuid", nargs=1, type=str, default="0")
    parser.add_argument("--backbone", action="store_true")
    parser.add_argument("-model", nargs=1, type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid[0]

    model_name = args.model[0]
    model = ".".join(model_name.split(".")[:-1])
    backbone_only = args.backbone

    model_path = os.path.join(model_dir, model_name)
    save_dir = os.path.join(experiment_dir, model)
    makedir(save_dir)

    ppnet = torch.load(model_path)
    ppnet = ppnet.cuda()

    true_labels = np.array([])
    predicted_labels = np.array([])

    # iterate over test data
    with torch.no_grad():
        for image, label in test_loader:
            input = image.cuda()
            target = label.cuda()

            if backbone_only:
                output = ppnet(input)
            else:
                output, _ = ppnet(input)

            _, predicted = torch.max(output.data, 1)
            true_labels = np.append(true_labels, label.numpy())
            predicted_labels = np.append(
                predicted_labels, predicted.cpu().numpy()
            )

    # Build confusion matrix
    cf_matrix = confusion_matrix(true_labels, predicted_labels)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
