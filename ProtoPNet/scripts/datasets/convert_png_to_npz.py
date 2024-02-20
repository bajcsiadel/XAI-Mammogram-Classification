import os
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import omegaconf.errors
from dotenv import load_dotenv
from icecream import ic
from tqdm import tqdm

load_dotenv()
sys.path.append(os.getenv("PROJECT_ROOT"))

from ProtoPNet.util.config_types import Config, init_config_store

conf_dir = (
    Path(os.getenv("PROJECT_ROOT"))
    / os.getenv("MODULE_NAME")
    / os.getenv("CONFIG_DIR_NAME")
)


@hydra.main(
    version_base=None,
    config_path=str(conf_dir),
    config_name="script_define_mean_config",
)
def convert_images(cfg: Config):
    images = [
        f
        for f in cfg.data.set.image_dir.iterdir()
        if f.is_file() and f.suffix == ".png"
    ]
    for filepath in tqdm(
        images,
        desc=f"Processing files for '{cfg.data.set.name}."
        f"{cfg.data.set.target.size}.{cfg.data.set.state}'",
    ):
        if filepath.is_file() and filepath.suffix == ".png":
            new_filepath = filepath.with_suffix(".npz")
            if not new_filepath.is_file():
                image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
                np.savez(new_filepath, image=image)
            os.remove(filepath)
            # print(f"Converted {filepath} to {new_filepath}")


init_config_store()

try:
    convert_images()
except omegaconf.errors.MissingMandatoryValue:
    ic("skipped")
