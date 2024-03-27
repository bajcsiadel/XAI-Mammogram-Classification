from . import densenet_features, resnet_features, vgg_features

all_features = (
    densenet_features.all_features
    | resnet_features.all_features
    | vgg_features.all_features
)
