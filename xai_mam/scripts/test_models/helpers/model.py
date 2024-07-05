import torch
from torch import nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self, n_classes: int, n_channels: int, use_dropouts: bool,
                 train_backbone: bool):
        super().__init__()
        self.features = resnet50(weights="IMAGENET1K_V2")

        if n_channels != 3:
            weights = self.features.conv1.weight.sum(dim=1, keepdim=True)
            self.features.conv1 = nn.Conv2d(
                n_channels, 64, 7, 2, 3, bias=False
            )
            self.features.conv1.weight = nn.Parameter(weights)

        resnet_feature_size = self.features.fc.in_features
        self.features.fc = nn.Identity()

        if use_dropouts:
            self.classification = nn.Sequential(
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(resnet_feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm1d(256),
                nn.Linear(256, n_classes),
            )
        else:
            self.classification = nn.Sequential(
                nn.Linear(resnet_feature_size, n_classes)
            )

        with torch.no_grad():
            for param in self.features.parameters():
                param.requires_grad = train_backbone

        self.train_backbone = train_backbone
        self.n_classes = n_classes
        self.in_channels = n_channels

    @property
    def out_channels(self):
        return self.n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classification(x)
