from . import densenet_features, resnet_features, vgg_features

BACKBONE_MODELS = [
    *densenet_features.model_urls.keys(),
    *resnet_features.model_urls.keys(),
    *vgg_features.model_urls.keys(),
]
