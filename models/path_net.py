"""
VGG19-BN Histopathology CNN (Path Network).

Architecture from Pathomic Fusion (Chen et al., 2022):
- VGG19 with batch normalization as feature extractor (ImageNet pretrained)
- Custom classifier head: 512*7*7 → 1024 → 1024 → path_dim
- Processes 512×512 histopathology ROIs at 20× magnification
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from utils.helpers import dfs_freeze


# ── VGG architecture configs ──────────────────────────────────────────

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# ── Helper ────────────────────────────────────────────────────────────

def make_layers(cfg, batch_norm=False):
    """Build VGG feature layers from config."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ── PathNet ───────────────────────────────────────────────────────────

class PathNet(nn.Module):
    """
    VGG19-BN based histopathology feature extractor.
    
    Outputs:
        features: (batch, path_dim) intermediate representations
        hazard: (batch, num_classes) final predictions
    """

    def __init__(self, features, path_dim=32, act=None, num_classes=1):
        super(PathNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act

        self.output_range = Parameter(
            torch.FloatTensor([6]), requires_grad=False
        )
        self.output_shift = Parameter(
            torch.FloatTensor([-3]), requires_grad=False
        )

        # Freeze pretrained VGG features
        dfs_freeze(self.features)

    def forward(self, **kwargs):
        x = kwargs['x_path']
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        hazard = self.linear(features)

        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard


# ── Factory ───────────────────────────────────────────────────────────

def get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True,
            label_dim=1, pretrained=True, progress=True, **kwargs):
    """
    Build a VGG19-BN PathNet with optional ImageNet pretraining.
    
    Args:
        arch: VGG variant name
        cfg: Layer configuration key
        act: Output activation
        batch_norm: Use batch normalization
        label_dim: Number of output classes
        pretrained: Load ImageNet weights
        
    Returns:
        PathNet model
    """
    model = PathNet(
        make_layers(cfgs[cfg], batch_norm=batch_norm),
        act=act, num_classes=label_dim, **kwargs
    )

    if pretrained:
        pretrained_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        # Remove classifier weights (we use our own head)
        for key in list(pretrained_dict.keys()):
            if 'classifier' in key:
                pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Loaded ImageNet pretrained VGG19-BN weights")

    return model
