"""
Pure a trained ProtoPNet model by pruning prototypes.
"""
import dataclasses as dc
import datetime
import pickle
import time
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import xai_mam.models.ProtoPNet._helpers.prune as prune
from xai_mam.utils.config.resolvers import resolve_run_location
from xai_mam.utils.config.types import BatchSize, Gpu, Outputs
from xai_mam.utils.environment import get_env
from xai_mam.utils.log import TrainLogger
from xai_mam.utils.preprocess import preprocess


@dc.dataclass
class ScriptConfig:
    result_dir: Path
    model_name: str
    optimize_last_layer: bool
    prune_threshold: int
    k_nearest: int
    outputs: Outputs
    batch_size: BatchSize
    gpu: Gpu = dc.field(default_factory=Gpu)


class PrototypesNotPushedError(Exception):
    ...


class BackboneOnlyError(Exception):
    ...


def epoch_from_model_name(model_name):
    """
    Define the epoch number from the name of the model.

    :param model_name: name of the model file
    :type model_name: str
    :return: epoch number
    :rtype: int
    """
    epoch_str = model_name.split("-push")[0]

    if "-" in epoch_str:
        return int(epoch_str.split("-")[0])

    return int(epoch_str)


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIG_PATH"),
    config_name="script_protopnet_run_pruning",
)
def main(cfg: ScriptConfig):
    with TrainLogger(
        "pruning", cfg.outputs, log_location=Path(cfg.result_dir)
    ) as logger:
        try:
            cfg = OmegaConf.to_object(cfg)

            with (cfg.result_dir / "config.pickle").open("rb") as f:
                original_config = pickle.load(f)

            # if original_config.model.backbone_only:
            #     raise BackboneOnlyError(f"The given experiment trains the "
            #                             f"backbone only\n{cfg.result_dir}")

            original_config.data.datamodule.cross_validation_folds = 1
            original_config.model.phases["finetune"].epochs = 100
            data_module = hydra.utils.instantiate(original_config.data.datamodule)

            model_location = logger.checkpoint_location / cfg.model_name

            if not model_location.exists():
                raise FileNotFoundError(f"Model file was not found at {model_location}")

            if "no_push" in cfg.model_name:
                raise PrototypesNotPushedError(
                    f"Specified model is before push, " f"but should be after."
                )

            epoch = epoch_from_model_name(cfg.model_name)

            trainer = original_config.model.params.construct_trainer(
                data_module=data_module,
                model_location=model_location,
                phases=original_config.model.phases,
                params=original_config.model.params,
                gpus=cfg.gpu,
                logger=logger,
            )

            train_loader = data_module.train_dataloader(batch_size=cfg.batch_size.train)
            validation_loader = data_module.validation_dataloader(
                batch_size=cfg.batch_size.validation
            )
            push_loader = data_module.push_dataloader(batch_size=cfg.batch_size.train)

            logger.info(f"train set size: {len(train_loader.dataset)}")
            logger.info(f"validation set size: {len(validation_loader.dataset)}")
            logger.info(f"push set size: {len(push_loader.dataset)}")
            logger.info(f"batch size:\n")
            logger.info(f"\ttrain: {train_loader.batch_size}")
            logger.info(f"\tvalidation: {validation_loader.batch_size}")
            logger.info(f"\tpush: {push_loader.batch_size}")

            trainer.eval(dataloader=validation_loader)

            logger.info("prune")
            start_time = time.time()
            prune.prune_prototypes(
                dataloader=push_loader,
                prototype_network_parallel=trainer.parallel_model,
                k=cfg.k_nearest,
                prune_threshold=cfg.prune_threshold,
                preprocess_input_function=preprocess,  # normalize
                original_model_dir=cfg.result_dir,
                epoch_number=epoch,
                # model_name=None,
                logger=logger,
                copy_prototype_imgs=True,
            )
            logger.info(
                f"finished pruning in "
                f"{datetime.timedelta(int(time.time() - start_time))}"
            )

            accuracy = trainer.eval(dataloader=push_loader)
            logger.save_model_w_condition(
                model_name=cfg.model_name.split("push")[0] + "prune",
                state=trainer.model.state_dict(),
                accu=accuracy,
                target_accu=0.70,
            )

            # last layer optimization
            if cfg.optimize_last_layer:
                trainer.last_layer(train_loader, validation_loader)

        except Exception as e:
            logger.exception(e)


resolve_run_location()
ConfigStore.instance().store("_script_config_validation", ScriptConfig)

main()
