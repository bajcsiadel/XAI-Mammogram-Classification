from . import densenet_features
from . import resnet_features
from . import vgg_features

BACKBONE_MODELS = [
    *densenet_features.model_urls.keys(),
    *resnet_features.model_urls.keys(),
    *vgg_features.model_urls.keys(),
]