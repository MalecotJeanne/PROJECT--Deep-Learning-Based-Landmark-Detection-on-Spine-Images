"""
Docstring:
"""

from monai.networks.nets import resnet18
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()
        self.model = resnet18(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        return self.model(x)