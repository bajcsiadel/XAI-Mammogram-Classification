from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.getenv("PROJECT_ROOT"))

from torch.utils.data import DataLoader
from ProtoPNet.dataset.metadata import DATASETS
from ProtoPNet.dataset.dataloaders import CustomVisionDataset

# --------------------------------------------- #
import argparse

parser = argparse.ArgumentParser(__file__, "Define the mean and std of a given dataset.")

parser.add_argument(
    "--dataset",
    type=str,
    choices=DATASETS.keys(),
    required=True,
    help="Dataset for which to define the mean and std."
)

args = parser.parse_args()
# --------------------------------------------- #

assert args.dataset in DATASETS.keys(), f"Dataset must be one of {DATASETS.keys()}, not {args.dataset}"

DS_META = DATASETS[args.dataset]

for image_type_name, image_type_details in DS_META.VERSIONS.items():
    print(f"IMAGE TYPE: {image_type_name}")
    for version_name, version in image_type_details.items():
        print(f"\tVERSION: {version_name}")
        DS_META.IMAGE_DIR = version.DIR
        dataset = CustomVisionDataset(
            DS_META,
            "normal_vs_abnormal" if image_type_name == "full" else "benign_vs_malignant",
            subset="all"
        )

        loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0, shuffle=False)

        images, _ = next(iter(loader))

        mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
        print(f"\t\tMEAN={mean.numpy()[0]:.4f},\n\t\tSTD={std.numpy()[0]:.4f},\n")
