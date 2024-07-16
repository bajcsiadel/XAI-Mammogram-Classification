"""
Convert images
==============

Convert all images in a directory to npz files. npz files are a bit
faster to load than png files.

Configuration
=============

File `xai_mam/conf/script_define_mean_config.yaml` contains the
configuration of the script.
"""
import dataclasses as dc
import os

import cv2
import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import errors as conf_errors
from tqdm import tqdm

from xai_mam.utils.config.resolvers import add_all_custom_resolvers
from xai_mam.utils.config.types import DatasetConfig
from xai_mam.utils.log import ScriptLogger


@dc.dataclass
class DataConfig:
    set: DatasetConfig


@dc.dataclass
class Config:
    data: DataConfig


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH"),
    config_name="script_convert_images_config",
)
def convert_images(cfg: Config):
    logger = ScriptLogger(__name__)

    try:
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
    except conf_errors.MissingMandatoryValue as e:
        logger.info(f"Dataset: {cfg['data']['set'].name}")
        logger.info(f"Size: {cfg['data']['set']['target'].size}")
        logger.info(f"State: {cfg['data']['set'].state}")
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


load_dotenv()
add_all_custom_resolvers()
config_store_ = DatasetConfig.init_store()
config_store_.store(name="_config_validation", node=Config)
config_store_.store(name="_data_set_validation", group="data/set", node=DatasetConfig)

convert_images()
