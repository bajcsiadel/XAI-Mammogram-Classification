from xai_mam.models.ProtoPNet._construct import construct_model, construct_trainer
from xai_mam.models.ProtoPNet._model import AddOnLayers
from xai_mam.models.ProtoPNet._model.backbone import ProtoPNetBackbone
from xai_mam.models.ProtoPNet._model.explainable import ProtoPNet
from xai_mam.models.ProtoPNet._trainer.backbone import BackboneTrainer
from xai_mam.models.ProtoPNet._trainer.explainable import ExplainableTrainer


def log_parameters(logger, cfg, number_of_classes):
    logger.info("")
    logger.info("prototype settings")
    with logger.increase_indent_context():
        logger.info(f"{cfg.model.params.prototypes.per_class} prototypes per class")
        logger.info(f"{cfg.model.params.prototypes.size} prototype size")
        cfg.model.params.prototypes.define_shape(number_of_classes)
        logger.info(
            f"{' x '.join(map(str, cfg.model.params.prototypes.shape))} prototype shape"
        )

    logger.info("")
    logger.info("network settings")
    with logger.increase_indent_context():
        logger.info(f"{cfg.model.network.name} backbone")
        logger.info(f"{cfg.model.network.add_on_layer_properties.type} "
                    f"add on layer type")
        logger.info(
            f"{cfg.model.network.add_on_layer_properties.activation} add on activation "
            f"({AddOnLayers.get_reference(cfg.model.network.add_on_layer_properties.activation)})"  # noqa
        )


def validate_model_config(cfg):
    if cfg.backbone_only:
        if "warm" in cfg.phases and cfg.phases["warm"].epochs != 0:
            raise ValueError(f"Training backbone does not support warmup")
        if "finetune" in cfg.phases and cfg.phases["warm"].epochs != 0:
            raise ValueError(f"Training backbone does not support warmup")
        if "push" in cfg.phases:
            raise ValueError(f"Training backbone does not support push phase")
