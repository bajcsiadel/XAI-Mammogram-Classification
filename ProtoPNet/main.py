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

from ProtoPNet.models.ProtoPNet import AddOnLayers
from ProtoPNet.utils import helpers
from ProtoPNet.utils.config import script_main as main_cfg
from ProtoPNet.utils.environment import get_env
from ProtoPNet.utils.log import Log

tick = "\u2714"
cross = "\u2718"


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH"),
    config_name="main_config",
)
def main(cfg: main_cfg.Config):
    with Log(__name__, cfg.outputs) as logger:
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
    if not gpu.disabled:
        match platform.system():
            case "Windows" | "Linux":
                logger.info(
                    f"\t{tick if torch.cuda.is_available() else cross} available CUDA"
                )
                logger.info(
                    f"\tVisible devices set to: {os.getenv('CUDA_VISIBLE_DEVICES')}"
                )
            case "Darwin":
                logger.info(
                    f"\t{tick if torch.backends.mps.is_available() else cross} available MPS"
                )

    else:
        logger.info(f"\t{tick} disabled")


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


def run_experiment(cfg: main_cfg.Config, logger: Log):
    # save last commit number (source of the used code)
    commit_file = logger.metadata_location / "commit_hash"
    commit_file.write_bytes(helpers.get_current_commit_hash())

    shutil.copy(cfg.data.set.metadata.file, logger.metadata_location)

    if cfg.cross_validation.folds > 1:
        logger.info("")
        logger.info(f"{tick} cross validation")
        logger.info(f"\t{cfg.cross_validation.folds} folds")
        logger.info(
            f"\t{tick if cfg.cross_validation.stratified else cross} stratified"
        )
        logger.info(f"\t{tick if cfg.cross_validation.balanced else cross} balanced")
        logger.info(f"\t{tick if cfg.cross_validation.grouped else cross} grouped")
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
    logger.info(f"\t{cfg.data.set.name}")
    logger.info(f"\t{cfg.data.set.state}")
    logger.info(f"\t{cfg.data.set.target.size}")
    logger.info(f"\t{cfg.data.set.target.name}")
    logger.info(f"\t{' x '.join(map(str, image_shape))} image shape")
    logger.info(f"\t{cfg.data.set.image_properties.color_channels} color channels")
    logger.info(f"\t{cfg.data.set.image_properties.std} std")
    logger.info(f"\t{cfg.data.set.image_properties.mean} mean")
    logger.info(f"\t{data_module.dataset.number_of_classes} classes")

    logger.info("")
    logger.info("prototype settings")
    logger.info(f"\t{cfg.model.params.prototypes.per_class} prototypes per class")
    logger.info(f"\t{cfg.model.params.prototypes.size} prototype size")
    cfg.model.params.prototypes.define_shape(data_module.dataset.number_of_classes)
    logger.info(
        f"\t{' x '.join(map(str, cfg.model.params.prototypes.shape))} prototype shape"
    )

    logger.info("")
    logger.info("network settings")
    logger.info(f"\t{cfg.model.network.name} backbone")
    logger.info(f"\t{cfg.model.network.add_on_layer_properties.type} add on layer type")
    logger.info(
        f"\t{cfg.model.network.add_on_layer_properties.activation} add on activation "
        f"({AddOnLayers.get_reference(cfg.model.network.add_on_layer_properties.activation)})"
    )
    logger.info(f"\t{tick if cfg.model.network.pretrained else cross} pretrained")
    logger.info(f"\t{tick if cfg.model.backbone_only else cross} backbone only")

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
            f"\t\tfold time: "
            f"{datetime.timedelta(seconds=int(time.time() - start_fold))}"
        )
        logger.info(f"\t\tfinished training fold {fold}")

    logger.info(
        f"training time: "
        f"{datetime.timedelta(seconds=int(time.time() - start_training))}"
    )
    logger.info("finished training")


if __name__ == "__main__":
    main_cfg.init_config_store()

    main()
