import argparse
import math
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils
from albumentations.pytorch import ToTensorV2
from database.dataloader import train_dir
from ProtoPNet.config.settings import (
    base_architecture,
    classes,
    experiment_run,
    img_shape,
    in_channels,
)
from ProtoPNet.util.helpers import makedir

experiment_dir = os.path.join(
    os.getcwd(), "saved_models", base_architecture, experiment_run
)
model_dir = os.path.join(experiment_dir, "model")

categories = []
if classes:
    categories = classes
else:
    categories = next(os.walk(train_dir))[1]
    categories.sort()


def read_misclassified_images_form_csv_for_class(
    file, true_class, num_examples_per_class
):
    df = pd.read_csv(file)
    df_cat = df[df["true_class"] == true_class]
    paths = df_cat["image_path"].to_numpy()
    predictions = df_cat["predicted_class"].to_numpy()
    images = torch.empty(
        (num_examples_per_class, in_channels, img_shape[0], img_shape[1])
    )
    augment = A.Compose(
        [
            A.ToFloat(max_value=4096),
            A.augmentations.geometric.resize.Resize(
                height=img_shape[0], width=img_shape[1]
            ),
            ToTensorV2(),
        ]
    )
    path_length = len(paths)
    limit = (
        num_examples_per_class
        if num_examples_per_class < path_length
        else path_length
    )
    for i in range(limit):
        image = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
        image = augment(image=image)["image"]
        images[i] = image
    return images, predictions


def create_collage_original_images(
    images, true_class, path, num_examples_per_class
):
    nrow = math.ceil(math.sqrt(num_examples_per_class))
    plt.figure(figsize=(15, 12))
    np_imagegrid = torchvision.utils.make_grid(
        images, nrow=nrow, pading=1
    ).numpy()
    if in_channels == 1:
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)), cmap="gray")
    else:
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
    plt.title(str(categories[true_class] + " (" + str(true_class) + ")"))
    plt.axis("off")
    plt.savefig(
        os.path.join(
            path, "misclassification_" + str(true_class) + "_original.png"
        ),
        orientation="landscape",
        bbox_inches="tight",
    )


def save_misclassification_example_original(
    image, prediction, true_class, count, path
):
    arr = np.transpose(image, (1, 2, 0))
    title = (
        categories[prediction]
        + " ("
        + str(prediction)
        + ") instead of "
        + categories[true_class]
        + " ("
        + str(true_class)
        + ")"
    )
    filename = (
        str(prediction)
        + "_instead_of_"
        + str(true_class)
        + "_"
        + str(count)
        + ".png"
    )
    fname = os.path.join(path, filename)
    plt.figure(figsize=(10, 8))
    plt.title(title)
    if in_channels == 1:
        plt.imshow(arr, cmap="gray")
    else:
        plt.imshow(arr)
    plt.axis("off")
    plt.savefig(fname=fname, orientation="landscape", bbox_inches="tight")


if __name__ == "__main__":
    ''' 
        Usage:  python extract_misclassified_images.py -model=''
                python generate_misclassification_collage.py -model='' -imclass 0 -numexamples 4
        - model: name of the model (.pth file)
        - imclass: number of a category for the dataset
        - numexamples: number of exampes to be saved 
        Generates heatmap collage for images from a given class
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", nargs=1, type=str)
    parser.add_argument("-imclass", nargs=1, type=int, default=0)
    parser.add_argument("-numexamples", nargs=1, type=int, default=4)
    args = parser.parse_args()

    model_name = args.model[0]
    model = ".".join(model_name.split(".")[:-1])
    misclass_dir = os.path.join(experiment_dir, model, "misclassification")
    imclass = args.imclass[0]
    num_examples_per_class = args.numexamples[0]
    csv_file = os.path.join(misclass_dir, "misclassified_images.csv")
    save_dir = os.path.join(misclass_dir, str(imclass))
    makedir(save_dir)

    images, predictions = read_misclassified_images_form_csv_for_class(
        csv_file, imclass, num_examples_per_class
    )
    create_collage_original_images(
        images, imclass, misclass_dir, num_examples_per_class
    )
    for i in range(num_examples_per_class):
        save_misclassification_example_original(
            images[i], predictions[i], imclass, i, save_dir
        )
