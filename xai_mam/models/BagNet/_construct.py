import torch

from xai_mam.models.BagNet._model.backbone import BagNetBackbone
from xai_mam.models.BagNet._model.explainable import all_models
from xai_mam.models.BagNet._trainer import BagNetTrainer
from xai_mam.models.utils.helpers import load_model


def construct_model(
    *,
    logger,
    base_architecture="bagnet17",
    pretrained=True,
    n_color_channels=3,
    n_classes=200,
    backbone_only=False,
    **kwargs,
):
    """
    Create a BagNet model (either backbone or explainable). Parameters should
    be specified as keyword arguments.

    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    :param base_architecture: backbone/feature network type. Defaults to ``"bagnet17"``.
    :type base_architecture: str
    :param pretrained: defines if the backbone is pretrained on ImageNet.
        Defaults to ``True``.
    :type pretrained: bool
    :param n_color_channels: number of color channels in the image. Defaults to ``3``.
    :type n_color_channels: int
    :param n_classes: number of classes in the dataset. Defaults to ``200``.
    :type n_classes: int
    :param backbone_only: defines if only the backbone is trained.
        Defaults to ``False``.
    :type backbone_only: bool
    :return: model instance
    :rtype: ProtoPNet.models.BagNet._model.BagNetBase
    """
    if backbone_only:
        return BagNetBackbone(
            n_classes=n_classes,
            logger=logger,
            n_color_channels=n_color_channels,
            pretrained=pretrained,
        )
    else:
        return all_models[base_architecture](
            n_classes=n_classes,
            logger=logger,
            n_color_channels=n_color_channels,
            pretrained=pretrained,
        )


def load_bagnet_model(model_location, logger):
    """
    Load a BagNet model from a file.

    :param model_location: location of the file in which the trained model is saved.
    :type model_location: pathlib.Path
    :param logger:
    :type logger: xai_mam.utils.log.TrainLogger
    :return: model instance
    :rtype: (xai_mam.models.BagNet._model.BagNetBase, dict)
    """
    return load_model(model_location, construct_model, logger)


def construct_trainer(
    *,
    data_module,
    gpu,
    logger,
    model_config=None,
    model_location=None,
    phases=None,
    params=None,
    fold=None,
    train_sampler=None,
    validation_sampler=None,
):
    """
    Create a trainer instance to train a xai_mam. Parameters should
    be specified as keyword arguments.

    :param data_module:
    :type data_module: ProtoPNet.dataset.dataloaders.CustomDataModule
    :param gpu: gpu usage information
    :type gpu: ProtoPNet.utils.config.types.Gpu
    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    :param model_config: configuration of the model. Defaults to ``None``.
    :type model_config: ProtoPNet.utils.config.types.ModelConfig | None
    :param model_location: location of the file in which the trained model is saved.
        Defaults to ``None``.
    :type model_location: str | pathlib.Path | None
    :param phases: phase details of the model. Defaults to ``None``.
    :type phases: dict[str, ProtoPNet.utils.config.types.Phase]
    :param params: parameters of the model. Defaults to ``None``.
    :type params: dict
    :param fold: Defaults to ``None``.
    :type fold: int | None
    :param train_sampler: Defaults to ``None``.
    :param validation_sampler: Defaults to ``None``.
    :return: trainer instance
    :rtype: ProtoPNet.models.ProtoPNet._trainer.ProtoPNetTrainer
    """
    if model_config is None and model_location is None:
        raise ValueError(
            f"Model not specified! Either 'model_location' "
            f"or 'model_config' must be specified."
        )
    if model_config is not None:
        n_classes = data_module.dataset.number_of_classes
        model_initialization_parameters = {
            "base_architecture": model_config.network.name,
            "pretrained": model_config.network.pretrained,
            "n_color_channels": data_module.dataset.image_properties.n_color_channels,
            "n_classes": n_classes,
            "backbone_only": model_config.backbone_only,
        }
        model = construct_model(
            logger=logger,
            **model_initialization_parameters,
        )
        if phases is None:
            phases = model_config.phases
        if params is None:
            params = model_config.params
    else:  # model_location is not None
        if phases is None or params is None:
            raise ValueError(
                f"If model is loaded from file 'phases' "
                f"and 'params' should be passed."
            )
        model, model_initialization_parameters = load_bagnet_model(model_location, logger)

    trainer_class = BagNetTrainer

    return trainer_class(
        fold=fold,
        data_module=data_module,
        train_sampler=train_sampler,
        validation_sampler=validation_sampler,
        model=model,
        phases=phases,
        params=params,
        loss=params.loss,
        gpu=gpu,
        model_initialization_parameters=model_initialization_parameters,
        logger=logger,
    )
