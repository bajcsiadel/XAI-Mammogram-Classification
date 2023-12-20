"""
Define the mean and std
=======================

Script to define the mean and std of a given image dataset.

Details in blog post [#]_.

References & Footnotes
======================

.. [#] https://www.binarystudy.com/2021/04/how-to-calculate-mean-standard-deviation-images-pytorch.html
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import hydra
from icecream import ic
import numpy as np
from omegaconf import OmegaConf, errors as conf_errors
import pipe
from torch.utils.data import DataLoader

load_dotenv()
sys.path.append(os.getenv("PROJECT_ROOT"))

from ProtoPNet.dataset.dataloaders import CustomVisionDataset, my_collate_function
from ProtoPNet.util import helpers
from ProtoPNet.util.config_types import Config


conf_dir = Path(os.getenv("PROJECT_ROOT")) / os.getenv("MODULE_NAME") / os.getenv("CONFIG_DIR_NAME")


def flatten(lst: list) -> np.ndarray:
    """
    Flatten a list of tensors to a numpy array

    :param lst:
    :return:
    """
    new_lst = []
    for tensor in lst:
        if len(new_lst) != tensor.shape[0]:
            new_lst = [new_lst] * tensor.shape[0]

        new_lst = np.hstack((new_lst, tensor.flatten(start_dim=1).tolist()))
    return new_lst


@hydra.main(version_base=None, config_path=str(conf_dir), config_name="script_define_mean_config")
def compute_mean_and_std_of_dataset(cfg: Config):
    logger = logging.getLogger()

    try:
        cfg = helpers.DotDict.new(OmegaConf.to_object(cfg))

        logger.info(f"Dataset: {cfg.data.set.name}")
        logger.info(f"Size: {cfg.data.set.size}")
        logger.info(f"State: {cfg.data.set.state}")
        logger.info(f"Target: {cfg.target}")

        dataset = CustomVisionDataset(
            cfg.data.set,
            cfg.target,
            subset="all",
            normalize=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=0,
            shuffle=False,
            collate_fn=my_collate_function,
        )

        average_image_size = np.zeros((2,))

        fst_moment = np.zeros((cfg.data.set.image_properties.color_channels,))
        snd_moment = np.zeros((cfg.data.set.image_properties.color_channels,))
        cnt = 0

        for images, _ in loader:

            sizes = list(images) | pipe.map(lambda x: x.shape[1:]) | helpers.CustomPipe.to_numpy
            average_image_size += np.sum(sizes, axis=0)

            images = flatten(images)

            sum_ = np.sum(images, axis=1)
            sum_of_square = np.sum(np.square(images), axis=1)

            fst_moment = (cnt * fst_moment + sum_) / (cnt + len(images[0]))
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + len(images[0]))
            cnt += len(images[0])

        mean, std = fst_moment, np.sqrt(snd_moment - fst_moment ** 2)

        average_image_size /= len(dataset)

        logger.info(f"Number of images: {len(dataset)}")
        logger.info(f"Average image size: {average_image_size}")

        logger.info(f"mean:")
        for m in mean:
            logger.info(f"- {m:.4f}")

        logger.info(f"std:")
        for s in std:
            logger.info(f"- {s:.4f}")

    except conf_errors.MissingMandatoryValue as e:
        logger.info(f"Dataset: {cfg['data']['set'].name}")
        logger.info(f"Size: {cfg['data']['set'].size}")
        logger.info(f"State: {cfg['data']['set'].state}")
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


compute_mean_and_std_of_dataset()
