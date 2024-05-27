import dataclasses as dc

import hydra
from hydra.core.config_store import ConfigStore
from icecream import ic

from xai_mam.utils.config._general_types import (
    CrossValidationParameters,
    DataConfig,
    Gpu,
    ModelConfig,
    Outputs,
)
from xai_mam.utils.config._general_types.data import init_data_config_store
from xai_mam.utils.config.resolvers import add_all_custom_resolvers
from xai_mam.utils.environment import get_env


@dc.dataclass
class JobProperties:
    number_of_workers: int

    def __setattr__(self, key, value):
        match key:
            case "number_of_workers":
                if value < 0:
                    raise ValueError(
                        f"Number of workers must be greater than 0.\n{key} = {value}"
                    )

        super().__setattr__(key, value)


@dc.dataclass
class Config:
    data: DataConfig
    cross_validation: CrossValidationParameters
    seed: int
    job: JobProperties
    outputs: Outputs
    model: ModelConfig
    gpu: Gpu = dc.field(default_factory=Gpu)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)


def init_config_store():
    from xai_mam.utils.config import config_store_

    add_all_custom_resolvers()

    config_store_.store(name="_config_validation", node=Config)
    init_data_config_store()
    config_store_.store(name="_model_validation", group="model", node=ModelConfig)
    config_store_.store(
        name="_cross_validation_validation",
        group="cross_validation",
        node=CrossValidationParameters,
    )

    return config_store_


@hydra.main(
    version_base=None, config_path=get_env("CONFIG_PATH"), config_name="main_config"
)
def process_config(cfg):
    """
    Process the information in the config file using hydra

    :param cfg: config information read from file
    :type cfg: Config
    :return: the processed and validated config
    """
    from hydra.utils import instantiate
    from omegaconf import omegaconf

    cfg: Config = omegaconf.OmegaConf.to_object(cfg)
    ic(cfg)

    ic(cfg.data.set.image_properties.augmentations)
    ic(instantiate(cfg.data.set.image_properties.augmentations.train.transforms))
    ic(type(cfg))
    ic(instantiate(cfg.data.datamodule))
    return cfg


if __name__ == "__main__":
    init_config_store()

    process_config()
