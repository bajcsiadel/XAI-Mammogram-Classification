import torch

from ProtoPNet.models.utils.backbone_features import all_features

from ._helpers.receptive_field import compute_proto_layer_rf_info_v2
from ._model.backbone import BBNet
from ._model.explainable import PPNet
from ._trainer.backbone import BackboneTrainer
from ._trainer.explainable import ExplainableTrainer


def construct_model(
    *,
    logger,
    base_architecture="resnet18",
    pretrained=True,
    color_channels=3,
    img_shape=(224, 224),
    prototype_shape=(2000, 512, 1, 1),
    n_classes=200,
    prototype_activation_function="log",
    add_on_layers_type="bottleneck",
    add_on_layers_activation="A",
    backbone_only=False,
    positive_weights_in_classifier=False,
):
    """
    Create a ProtoPNet model (either backbone or explainable). Parameters should
    be specified as keyword arguments.

    :param logger:
    :type logger: ProtoPNet.utils.log.Log
    :param base_architecture: backbone/feature network type. Defaults to ``"resnet18"``.
    :type base_architecture: str
    :param pretrained: defines if the backbone is pretrained on ImageNet.
        Defaults to ``True``.
    :type pretrained: bool
    :param color_channels: number of color channels in the image. Defaults to ``3``.
    :type color_channels: int
    :param img_shape: shape of the input image. Defaults to ``(224, 224)``.
    :type img_shape: tuple[int, int]
    :param prototype_shape: shape of the prototype features.
        Defaults to ``(2000, 512, 1, 1)``.
    :type prototype_shape: tuple[int, int, int, int]
    :param n_classes: number of classes in the dataset. Defaults to ``200``.
    :type n_classes: int
    :param prototype_activation_function: Defaults to ``"log"``.
    :type prototype_activation_function: str
    :param add_on_layers_type: layers added between the feature layer and
        the classification.Defaults to ``"bottleneck"``.
    :param add_on_layers_activation: activation of the add-on layers.
        Defaults to ``"A"``.
    :param backbone_only: defines if only the backbone is trained.
        Defaults to ``False``.
    :type backbone_only: bool
    :param positive_weights_in_classifier:
    :type positive_weights_in_classifier: bool
    :return: model instance
    :rtype: ProtoPNet.models.ProtoPNet._model.ProtoPNetBase
    """
    features = all_features[base_architecture].construct(
        pretrained=pretrained,
        color_channels=color_channels,
    )
    if backbone_only:
        return BBNet(
            features=features,
            img_shape=img_shape,
            prototype_shape=prototype_shape,
            n_classes=n_classes,
            logger=logger,
            color_channels=color_channels,
            add_on_layers_type=add_on_layers_type,
            add_on_layers_activation=add_on_layers_activation,
        )
    else:
        (
            layer_filter_sizes,
            layer_strides,
            layer_paddings,
        ) = features.conv_info()
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=img_shape[0],  # TODO img_size to img_shape
            layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides,
            layer_paddings=layer_paddings,
            prototype_kernel_size=prototype_shape[2],
        )
        return PPNet(
            features=features,
            img_shape=img_shape,
            prototype_shape=prototype_shape,
            proto_layer_rf_info=proto_layer_rf_info,
            n_classes=n_classes,
            logger=logger,
            init_weights=True,
            prototype_activation_function=prototype_activation_function,
            add_on_layers_type=add_on_layers_type,
            add_on_layers_activation=add_on_layers_activation,
            positive_weights_in_classifier=positive_weights_in_classifier,
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
    Create a trainer instance to train a ProtoPNet. Parameters should
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
        image_shape = data_module.dataset.input_size
        model = construct_model(
            logger=logger,
            base_architecture=model_config.network.name,
            pretrained=model_config.network.pretrained,
            color_channels=data_module.dataset.image_properties.color_channels,
            img_shape=image_shape,
            prototype_shape=model_config.params.prototypes.shape,
            n_classes=n_classes,
            prototype_activation_function=model_config.params.prototypes.activation_fn,
            add_on_layers_type=model_config.network.add_on_layer_properties.type,
            add_on_layers_activation=model_config.network.add_on_layer_properties.activation,
            backbone_only=model_config.backbone_only,
            positive_weights_in_classifier=False,
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
        model = torch.load(model_location)

    if model.backbone_only:
        trainer_class = BackboneTrainer
    else:
        trainer_class = ExplainableTrainer

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
