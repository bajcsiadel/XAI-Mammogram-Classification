import datetime
import os
import pickle
import platform
import shutil
import time
import warnings

import hydra
import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate

from xai_mam.models.ProtoPNet import AddOnLayers
from xai_mam.utils import helpers
from xai_mam.utils.config import script_main as main_cfg
from xai_mam.utils.environment import get_env
from xai_mam.utils.log import TrainLogger

tick = "\u2714"
cross = "\u2718"


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH"),
    config_name="main_config",
)
def main(cfg: main_cfg.Config):
    with TrainLogger(__name__, cfg.outputs) as logger:
        try:
            warnings.showwarning = lambda message, *args: logger.exception(
                message, warn_only=True
            )

            logger.log_command_line()

            cfg = omegaconf.OmegaConf.to_object(cfg)

            set_seeds(cfg.seed)

            log_gpu_usage(cfg.gpu, logger)
            if cfg.gpu.disabled:
                cfg.gpu.device = "cpu"

            run_experiment(cfg, logger)
        except Exception as e:
            logger.exception(e)


def log_gpu_usage(gpu, logger):
    """
    Log the gpu usage according to the given configuration.

    :param gpu:
    :type gpu: ProtoPNet.utils.config.types.Gpu
    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    :return:
    """
    logger.info("GPU settings")
    logger.increase_indent()
    if not gpu.disabled:
        match platform.system():
            case "Windows" | "Linux":
                logger.info(
                    f"{tick if torch.cuda.is_available() else cross} available CUDA"
                )
                logger.info(
                    f"Visible devices set to: {os.getenv('CUDA_VISIBLE_DEVICES')}"
                )
            case "Darwin":
                logger.info(
                    f"{tick if torch.backends.mps.is_available() else cross} available MPS"
                )

    else:
        logger.info(f"{tick} disabled")
    logger.decrease_indent()


def set_seeds(seed):
    """
    Set the seeds for all used libraries to the given value.
    :param seed:
    :type seed: int | None
    :return:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_experiment(cfg: main_cfg.Config, logger: Logger):
    # save last commit number (source of the used code)
    commit_file = logger.metadata_location / "commit_hash"
    commit_file.write_bytes(helpers.get_current_commit_hash())

    shutil.copy(cfg.data.set.metadata.file, logger.metadata_location)

    if cfg.cross_validation.folds > 1:
        logger.info("")
        logger.info(f"{tick} cross validation")
        logger.increase_indent()
        logger.info(f"{cfg.cross_validation.folds} folds")
        logger.info(
            f"{tick if cfg.cross_validation.stratified else cross} stratified"
        )
        logger.info(f"{tick if cfg.cross_validation.balanced else cross} balanced")
        logger.info(f"{tick if cfg.cross_validation.grouped else cross} grouped")
        logger.decrease_indent()
    else:
        logger.info(f"{cross} cross validation")

    data_module = instantiate(cfg.data.datamodule)

    logger.info("")
    logger.info(f"{tick if data_module.debug else cross} debug")

    image_shape = (
        cfg.data.set.image_properties.height,
        cfg.data.set.image_properties.width,
    )

    logger.info("")
    logger.info("data settings")
    logger.increase_indent()
    logger.info(f"{cfg.data.set.name}")
    logger.info(f"{cfg.data.set.state}")
    logger.info(f"{cfg.data.set.target.size}")
    logger.info(f"{cfg.data.set.target.name}")
    logger.info(f"{' x '.join(map(str, image_shape))} image shape")
    logger.info(f"{cfg.data.set.image_properties.color_channels} color channels")
    logger.info(f"{cfg.data.set.image_properties.std} std")
    logger.info(f"{cfg.data.set.image_properties.mean} mean")
    logger.info(f"{data_module.dataset.number_of_classes} classes")

    hydra.utils.instantiate(cfg.model.log_parameters_fn)(number_of_classes=data_module.dataset.number_of_classes, logger=logger, cfg=cfg)
    with logger.increase_indent_context():
        logger.info(f"{tick if cfg.model.network.pretrained else cross} pretrained")
        logger.info(f"{tick if cfg.model.backbone_only else cross} backbone only")
    logger.decrease_indent()

    with (logger.log_location / get_env("CONFIG_DIR_NAME") / "config.pickle").open(
        "wb"
    ) as f:
        pickle.dump(cfg, f)

    # train the model
    logger.info("start training")
    start_training = time.time()

    for fold, (train_sampler, validation_sampler) in enumerate(
        data_module.folds, start=1
    ):
        if cfg.cross_validation.folds > 1:
            logger.increase_indent()
        start_fold = time.time()
        trainer = cfg.model.params.construct_trainer(
            data_module=data_module,
            model_config=cfg.model,
            gpu=cfg.gpu,
            logger=logger,
            fold=fold,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
        )

        trainer.execute()

        del trainer

        logger.info(
            f"fold time: "
            f"{datetime.timedelta(seconds=int(time.time() - start_fold))}"
        )
        logger.info(f"finished training fold {fold}")
        logger.decrease_indent()

    logger.info(
        f"training time: "
        f"{datetime.timedelta(seconds=int(time.time() - start_training))}"
    )
    logger.info("finished training")


if __name__ == "__main__":
    main_cfg.init_config_store()

    main()
