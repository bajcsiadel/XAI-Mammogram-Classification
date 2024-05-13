import torch

from xai_mam.models.BagNet._model.backbone import BagNetBackbone
from xai_mam.models.BagNet._model.explainable import all_models
from xai_mam.models.BagNet._trainer import BagNetTrainer


def construct_model(
    *,
    logger,
    base_architecture="bagnet17",
    pretrained=True,
    color_channels=3,
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
    :param color_channels: number of color channels in the image. Defaults to ``3``.
    :type color_channels: int
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
            color_channels=color_channels,
            pretrained=pretrained,
        )
    else:
        return all_models[base_architecture](
            n_classes=n_classes,
            logger=logger,
            color_channels=color_channels,
            pretrained=pretrained,
        )


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
        model = construct_model(
            logger=logger,
            base_architecture=model_config.name,
            pretrained=model_config.network.pretrained,
            color_channels=data_module.dataset.image_properties.color_channels,
            n_classes=n_classes,
            backbone_only=model_config.backbone_only,
        )
        if phases is None:
            phases = model_config.phases
        if params is None:
            params = model_config.params
    else:  # model_location is not None
        if not model_location.exists():
            raise FileNotFoundError(f"Model file not found at {model_location}")
        if phases is None or params is None:
            raise ValueError(
                f"If model is loaded from file 'phases' "
                f"and 'params' should be passed."
            )
        # TODO: correct it. checkpoint contains:
        # model.state_dict()
        # optimizer.state_dict()
        # scheduler.state_dict()
        model = torch.load(model_location)

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
        logger=logger,
    )
