import os
import torch
from collections import OrderedDict
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ResNet3D(nn.Module):
    def __init__(self):
        super(resnet3d, self).__init__()

        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    def forward(self, x):
        print("x shape:", x.shape)
        output = self.model(x)
        print("output:", output.shape)

        return output
