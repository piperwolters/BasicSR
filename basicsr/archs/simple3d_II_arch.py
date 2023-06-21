import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Simple3DConvNetII(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=512, num_block=23, num_grow_ch=64):
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
        print("input:", x.shape)
        out = self.conv1(x)
        print("Conv1:", out.shape)
        out = self.pool(self.relu(out))
        print("Pool1:", out.shape)
        
        out = self.conv15(out)
        print("Conv 1.5:", out.shape)

        out = self.conv2(out)
        print("Conv2:", out.shape)
        out = self.pool(self.relu(out))
        print("Pool2:", out.shape)
        out = self.conv25(out)
        print("Conv 2.5:", out.shape)

        out = self.conv3(out)
        print("Conv3:", out.shape)
        out = self.pool(self.relu(out))
        print("Pool3:", out.shape)
        out = self.conv35(out)
        print("Conv 3.5:", out.shape)

        out = self.conv4(out)
        print("Conv4:", out.shape))
        out = self.pool(self.relu(out))
        print("Pool4:", out.shape)
        out = self.conv45(out)
        print("Conv 4.5:", out.shape)

        out = out.squeeze(2)
        print("Temporal Dim Squeeze:", out.shape)

        up = self.up1(out)
        print("Up1:", up.shape)
        up = self.up2(up)
        print("Up2:", up.shape)
        up = self.up3(up)
        print("Up3:", up.shape)

        return up

