import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Simple3DConvNetII(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=64):
        super(Simple3DConvNetII, self).__init__()

        self.conv1 = nn.Conv3d(num_in_ch, num_feat, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv15 = nn.Conv3d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(num_feat, num_feat*2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv25 = nn.Conv3d(num_feat*2, num_feat*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(num_feat*2, num_feat*4, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv35 = nn.Conv3d(num_feat*4, num_feat*4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(num_feat*4, num_feat*8, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv45 = nn.Conv3d(num_feat*8, num_feat*8, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.up1 = nn.ConvTranspose2d(num_feat*8, num_feat*4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(num_feat*4, num_feat*2, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(num_feat*2, 3, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(self.relu(out))
        
        out = self.conv15(out)

        out = self.conv2(out)
        out = self.pool(self.relu(out))
        out = self.conv25(out)

        out = self.conv3(out)
        out = self.pool(self.relu(out))
        out = self.conv35(out)

        out = self.conv4(out)
        out = self.pool(self.relu(out))
        out = self.conv45(out)

        out = out.squeeze(2)

        up = self.up1(out)
        up = self.up2(up)
        up = self.up3(up)

        return up

