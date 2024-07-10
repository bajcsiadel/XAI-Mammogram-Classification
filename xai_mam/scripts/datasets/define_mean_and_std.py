"""
Define the mean and std
=======================

Script to define the mean and std of a given image dataset.

Details in blog post [#]_.

References & Footnotes
======================

.. [#] https://www.binarystudy.com/2021/04/how-to-calculate-mean-standard-deviation-images-pytorch.html

Configuration
=============

File `xai_mam/conf/script_define_mean_config.yaml` contains the
configuration of the script.
"""
import dataclasses as dc
import os
import typing as typ

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf
from omegaconf import errors as conf_errors
from torch.utils.data import DataLoader

from xai_mam.dataset.dataloaders import my_collate_function
from xai_mam.utils import custom_pipe
from xai_mam.utils.config._general_types.data import DatasetConfig
from xai_mam.utils.config.resolvers import add_all_custom_resolvers
from xai_mam.utils.log import ScriptLogger


@dc.dataclass
class DataConfig:
    set: DatasetConfig


@dc.dataclass
class Config:
    data: DataConfig
    dataset: dict[str, typ.Any]


def flatten(lst):
    """
    Flatten a list of tensors to a numpy array

    :param lst:
    :type lst: list[torch.Tensor]
    :return:
    :rtype: numpy.ndarray
    """
    new_lst = []
    for tensor in lst:
        if len(new_lst) != tensor.shape[0]:
            new_lst = [new_lst] * tensor.shape[0]

        new_lst = np.hstack((new_lst, tensor.flatten(start_dim=1).tolist()))
    return new_lst


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH"),
    config_name="script_define_mean_config",
)
def compute_mean_and_std_of_dataset(cfg: Config):
    logger = ScriptLogger(__name__)

    try:
        cfg = OmegaConf.to_object(cfg)

        logger.info(f"Dataset: {cfg.data.set.name}")
        logger.info(f"Size: {cfg.data.set.target.size}")
        logger.info(f"State: {cfg.data.set.state}")
        logger.info(f"Target: {cfg.data.set.target.name}")

        dataset = hydra.utils.instantiate(cfg.dataset)

        loader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=0,
            shuffle=False,
            collate_fn=my_collate_function,
        )

        average_image_size = np.zeros((2,))

        fst_moment = np.zeros((cfg.data.set.image_properties.n_color_channels,))
        snd_moment = np.zeros((cfg.data.set.image_properties.n_color_channels,))
        cnt = 0

        for images, _ in loader:
            sizes = (
                list(images)
                | custom_pipe.map(lambda x: x.shape[1:])
                | custom_pipe.to_numpy
            )
            average_image_size += np.sum(sizes, axis=0)

            images = flatten(images)

            sum_ = np.sum(images, axis=1)
            sum_of_square = np.sum(np.square(images), axis=1)

            fst_moment = (cnt * fst_moment + sum_) / (cnt + len(images[0]))
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + len(images[0]))
            cnt += len(images[0])

        mean, std = fst_moment, np.sqrt(snd_moment - fst_moment**2)

        average_image_size /= len(dataset)

        logger.info(f"Number of images: {len(dataset)}")
        logger.info(f"Average image size: {average_image_size}")

        logger.info("mean:")
        for m in mean:
            logger.info(f"- {m:.4f}")

        logger.info("std:")
        for s in std:
            logger.info(f"- {s:.4f}")

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
config_store_.store(name="_data_validation", group="data", node=DataConfig)
compute_mean_and_std_of_dataset()
