"""
Augment images
==============

Apply augmentations to images in a directory and save them.

Configuration
=============

File `xai_mam/conf/script_define_mean_config.yaml` contains the
configuration of the script.
"""
import dataclasses as dc
import os
import shutil
import sys
import typing as typ
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf import errors as conf_errors
from tqdm import tqdm

load_dotenv()
sys.path.append(os.getenv("PROJECT_ROOT"))

from xai_mam.utils.config import config_store_
from xai_mam.utils.config._general_types.data import Dataset, Augmentations
from xai_mam.utils.config.resolvers import add_all_custom_resolvers
from xai_mam.utils.log import ScriptLogger


@dc.dataclass
class Data:
    set: Dataset


@dc.dataclass
class Config:
    data: Data
    dataset: dict[str, typ.Any]
    augmentations: Augmentations
    output_dir: Path

    def __post_init__(self):
        if not self.output_dir.is_absolute():
            self.output_dir = self.data.set.image_dir / self.output_dir

        self.output_dir.mkdir(exist_ok=True)


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH"),
    config_name="script_augment_config",
)
def augment_images(cfg: Config):
    logger = ScriptLogger(__name__)

    try:
        cfg = OmegaConf.to_object(cfg)

        augmented_data = pd.DataFrame(
            columns=["image_name", "original_image", "augmentation"]
        )

        import albumentations as A

        dataset = hydra.utils.instantiate(cfg.dataset)

        transforms = hydra.utils.instantiate(cfg.augmentations)

        for index, row in tqdm(dataset.metadata.iterrows(), desc="Images:"):
            image_path = (
                cfg.data.set.image_dir
                / f"{index[1]}{cfg.data.set.image_properties.extension}"
            )
            image = (
                np.load(image_path, allow_pickle=True)["image"]
                if image_path.suffix in [".npy", ".npz"]
                else cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            )

            augmented_image_path = (cfg.output_dir
                                    / image_path.with_stem(f"{image_path.stem}_0").name)
            shutil.copy(
                image_path,
                cfg.output_dir / image_path.with_stem(f"{image_path.stem}_0").name
            )
            augmented_data[len(augmented_data)] = [
                augmented_image_path, image_path, "original"
            ]
            count = 1
            for transform in transforms.train:
                augmented_images = transform(image=image)
                if type(augmented_images) is dict:
                    augmented_images = [augmented_images]

                for augmented_image in augmented_images:
                    new_image = augmented_image["image"]
                    augmented_image_path = (cfg.output_dir
                                            / image_path.with_stem(f"{image_path.stem}"
                                                                   f"_{count}").name)
                    # if augmented_image_path.suffix == ".npz":
                    np.savez(augmented_image_path, image=new_image)
                    # else:
                    cv2.imwrite(str(augmented_image_path.with_suffix(".png")), new_image)
                    augmented_data[len(augmented_data)] = [
                        augmented_image_path, image_path, transform
                    ]
                    count += 1
    except conf_errors.MissingMandatoryValue as e:
        logger.info(f"Dataset: {cfg['data']['set'].name}")
        logger.info(f"Size: {cfg['data']['set']['target'].size}")
        logger.info(f"State: {cfg['data']['set'].state}")
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


add_all_custom_resolvers()
config_store_.store(name="_config_validation", node=Config)
config_store_.store(name="_data_validation", group="data", node=Data)
config_store_.store(name="_data_set_validation", group="data/set", node=Dataset)
augment_images()
