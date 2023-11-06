from dotenv import load_dotenv
load_dotenv()

import cv2
import os
import numpy as np
from tqdm import tqdm

from ProtoPNet.dataset.metadata import DATASETS

DS_NAME = "DDSM"

DS_META = DATASETS[DS_NAME]

for version in tqdm(DS_META.VERSIONS.values(), desc="Processing directories"):
    directory = version.DIR
    for filename in tqdm(os.listdir(directory), desc="Processing files"):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            np.savez(filepath[:-4], image=image)
            # os.remove(filepath)
            # print(f"Converted {filepath} to {filepath[:-4]}.npz")
