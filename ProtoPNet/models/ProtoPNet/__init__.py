from ._construct import construct_model, construct_trainer
from ._model import AddOnLayers
from ._model.backbone import ProtoPNetBackbone
from ._model.explainable import ProtoPNet
from ._trainer.backbone import BackboneTrainer
from ._trainer.explainable import ExplainableTrainer


def log_parameters(logger, cfg, number_of_classes):
    logger.info("")
    logger.info("prototype settings")
    logger.info(f"\t{cfg.model.params.prototypes.per_class} prototypes per class")
    logger.info(f"\t{cfg.model.params.prototypes.size} prototype size")
    cfg.model.params.prototypes.define_shape(number_of_classes)
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
