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
from xai_mam.utils.config.resolvers import add_all_custom_resolvers
from xai_mam.utils.environment import get_env


@dc.dataclass
class JobConfig:
    n_workers: int

    def __setattr__(self, key, value):
        match key:
            case "n_workers":
                if value < 0:
                    raise ValueError(
                        f"Number of workers must be greater than 0.\n{key} = {value}"
                    )

        super().__setattr__(key, value)

    @staticmethod
    def init_store(
        config_store_: ConfigStore = None, group: str = "job"
    ) -> ConfigStore:
        if config_store_ is None:
            from xai_mam.utils.config import config_store_

        config_store_.store(name="_job_config_validation", group=group, node=JobConfig)

        return config_store_


@dc.dataclass
class Config:
    data: DataConfig
    cross_validation: CrossValidationParameters
    seed: int
    job: JobConfig
    outputs: Outputs
    model: ModelConfig
    gpu: Gpu = dc.field(default_factory=Gpu)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    @staticmethod
    def init_store():
        from xai_mam.utils.config import config_store_

        add_all_custom_resolvers()

        config_store_.store(name="_config_validation", node=Config)
        DataConfig.init_store(config_store_)
        ModelConfig.init_store(config_store_)
        CrossValidationParameters.init_config(config_store_)


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
    Config.init_config_store()

    process_config()
