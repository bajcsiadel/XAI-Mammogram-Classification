import warnings

from xai_mam.models.BagNet._construct import construct_model, construct_trainer
from xai_mam.models.BagNet._model.explainable import all_models


def log_parameters(logger, cfg, **kwargs):
    logger.info("")
    logger.info("network settings")
    with logger.increase_indent_context():
        logger.info(f"{cfg.model.network.name} backbone")


def validate_model_config(cfg):
    if cfg.backbone_only:
        if cfg.network.name != "resnet50":
            warnings.warn("BagNet backbone only supports 'resnet50'. "
                          "Changing to 'resnet50'.")
            cfg.network.name = "resnet50"
    else:
        if cfg.network.name not in all_models.keys():
            raise ValueError(f"Model {cfg.network.name} not supported for explainable "
                             f"BagNet. Choose one of {', '.join(all_models.keys())}.")
