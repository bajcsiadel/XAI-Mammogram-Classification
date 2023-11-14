from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.getenv("PROJECT_ROOT"))

import cv2
import numpy as np
from tqdm import tqdm

from ProtoPNet.dataset.metadata import DATASETS

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

DS_META = DATASETS[args.dataset]

for image_type_name, image_type_details in tqdm(DS_META.VERSIONS.items(), desc="Processing directories"):
    for version_name, version in tqdm(image_type_details.items(), desc=f"\tProcessing '{image_type_name}'"):
        directory = version.DIR
        for filename in tqdm(os.listdir(directory), desc=f"\t\tProcessing files for '{version_name}'"):
            if filename.endswith(".png"):
                filepath = os.path.join(directory, filename)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                np.savez(filepath[:-4], image=image)
                # os.remove(filepath)
                # print(f"Converted {filepath} to {filepath[:-4]}.npz")
