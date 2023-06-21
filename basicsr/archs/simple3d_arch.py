import os
import torch
from collections import OrderedDict
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Simple3DConvNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(Simple3DConvNet, self).__init__()

        self.conv1 = nn.Conv3d(num_in_ch, num_feat, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(num_feat+num_grow_ch, num_in_ch+num_grow_ch*2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv3 = nn.Conv3d(num_feat+num_grow_ch*2, num_feat+num_grow_ch*3, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(num_feat+num_grow_ch*3, num_feat+num_grow_ch*4, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))

        self.fc = nn.Linear(32 * 8 * 8 * 8, num_out_ch)

    def forward(self, x):
        print("x:", x.shape)
        out = self.conv1(x)
        print("Conv1 output shape:", out.shape)
        out = self.pool(self.relu(out))
        print("Pool1 output shape:", out.shape)
        out = self.conv2(out)
        print("Conv2 output shape:", out.shape)
        out = self.pool(self.relu(out))
        print("Pool2 output shape:", out.shape)
        out = self.conv3(out)
        print("Conv3:", out.shape)
        out = self.pool(self.relu(out))
        print("Pool3 output shape:", out.shape)
        out = self.conv4(out)
        print("conv4:", out.shape)
        out = self.pool(self.relu(out))
        print("Pool4 output shape:", out.shape)
        out = out.view(out.size(0), -1)
        print("out:", out.shape)
        out = self.fc(out)
        print("Final output shape:", out.shape)
        return out

