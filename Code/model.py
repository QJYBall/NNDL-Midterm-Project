import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Block(nn.Module):

    def __init__(self, in_chs, out_chs, mid_chs=None):
        super().__init__()

        if not mid_chs:
            mid_chs = out_chs

        self.Conv = nn.Sequential(
            nn.Conv2d(in_chs, mid_chs, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(mid_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Conv(x)
        return out


class Down(nn.Module):

    def __init__(self, in_chs, out_chs):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_Block(in_chs, out_chs)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return out


class Up(nn.Module):

    def __init__(self, in_chs, out_chs, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Conv_Block(in_chs, out_chs, in_chs // 2)
        else:
            self.up = nn.ConvTranspose2d(in_chs , in_chs // 2, kernel_size=2, stride=2)
            self.conv = Conv_Block(in_chs, out_chs)


    def forward(self, x_old, x_new):
        x_new = self.up(x_new)

        diffY = x_old.size()[2] - x_new.size()[2]
        diffX = x_old.size()[3] - x_new.size()[3]

        x_new = F.pad(x_new, [diffX // 2, diffX - diffX // 2, 
                              diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_old, x_new], dim=1)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class Recurent_Block(nn.module):
    def __init__(self, in_chs, out_chs):
        super(Recurent_Block, self).__init__()
        
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.InConv = Conv_Block(n_channels, 64)

        self.Down1 = Down(64, 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 1024 // factor)

        self.Up1 = Up(1024, 512 // factor, bilinear)
        self.Up2 = Up(512, 256 // factor, bilinear)
        self.Up3 = Up(256, 128 // factor, bilinear)
        self.Up4 = Up(128, 64, bilinear)

        self.OutConv = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.InConv(x)

        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)

        x = self.Up1(x4, x5)
        x = self.Up2(x3, x)
        x = self.Up3(x2, x)
        x = self.Up4(x1, x)

        out = self.OutConv(x)
        return out
