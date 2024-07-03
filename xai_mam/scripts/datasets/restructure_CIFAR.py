"""
Restructure CIFAR
=================

Restructure the CIFAR-10 dataset to be compatible with the XAI-MAM datamodule.
"""
import pickle

import pandas as pd
import numpy as np

from pathlib import Path

from dotenv import load_dotenv

from xai_mam.utils.environment import get_env


def process_batch(batch_file, df, meta, output_paths):
    subset = "test" if "test" in batch_file.stem else "train"

    batch = read_binary_file(batch_file, encoding="bytes")
    for image, label in zip(batch[b"data"], batch[b"labels"], strict=True):
        image = image.reshape(3, 32, 32)
        filename = f"{meta['label_names'][label]}_{len(df) + 1:05}"
        for d in output_paths:
            np.savez(d[0] / d[1].format(filename), image=image)
        df.loc[(len(df) + 1, filename), :] = [label, subset, label, subset, label, subset]
    return df


def read_binary_file(file_location, encoding="ASCII"):
    with open(file_location, "rb") as f:
        return pickle.load(f, encoding=encoding)


if __name__ == "__main__":
    load_dotenv()
    DATA_ROOT = get_env("DATA_ROOT")
    path = Path(DATA_ROOT, "CIFAR-10")
    output_paths = [(path / "images", "{}.npz")]

    cifar_meta = read_binary_file(path / "batches.meta")

    metadata = pd.DataFrame(
        columns=pd.MultiIndex.from_product([
            [
                "normal_vs_abnormal",
                "benign_vs_malignant",
                "normal_vs_benign_vs_malignant"
            ],
            ["label", "subset"]
        ]),
        index=pd.MultiIndex([[], []], [[], []], names=["patient_id", "image_name"])
    )
    for file in path.glob("*_batch*"):
        metadata = process_batch(file, metadata, cifar_meta, output_paths)
    metadata.to_csv(path / "extended_data.csv")
