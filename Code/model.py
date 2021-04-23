import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Block(nn.Module):

    def __init__(self, in_chs, out_chs, mid_chs=None):
        super(Conv_Block, self).__init__()

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


class Recurrent_Block(nn.Module):

    def __init__(self, out_chs, t=2):
        super(Recurrent_Block,self).__init__()

        self.t = t
        self.ch_out = out_chs
        self.conv = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.BatchNorm2d(out_chs),
			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i==0:
                x_ = self.conv(x)
            x_ = self.conv(x + x_)
        return x_


class RRCNN_Block(nn.Module):

    def __init__(self, in_chs, out_chs, t=2):
        super(RRCNN_Block,self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_Block(out_chs,t=t),
            Recurrent_Block(out_chs,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x_ = self.RCNN(x)
        return x + x_


class Attention_Block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_Block,self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x * psi


class OutConv(nn.Module):

    def __init__(self, in_chs, out_chs):
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class UNet_Down(nn.Module):

    def __init__(self, in_chs, out_chs, model="UNet"):
        super(UNet_Down, self).__init__()

        self.model = model

        if self.model=="UNet" or self.model=="Attention_UNet":
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                Conv_Block(in_chs, out_chs)
            )
        elif self.model=="R2UNet" or self.model=="Attention_R2UNet":
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                RRCNN_Block(in_chs, out_chs)
            )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return out


class UNet_Up(nn.Module):

    def __init__(self, in_chs, out_chs, bilinear=True, model="UNet"):
        super(UNet_Up, self).__init__()

        self.model = model

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            if self.model=="Attention_UNet" or self.model=="Attention_R2UNet":
                self.attention = Attention_Block(F_g=out_chs, F_l=out_chs, F_int=out_chs//2)
            
            if self.model=="UNet" or self.model=="Attention_UNet":
                self.conv = Conv_Block(in_chs, out_chs, in_chs // 2)
            elif self.model=="R2UNet" or self.model=="Attention_R2UNet":
                self.conv = RRCNN_Block(in_chs, out_chs)
        else:
            self.up = nn.ConvTranspose2d(in_chs , in_chs // 2, kernel_size=2, stride=2)

            if self.model=="Attention_UNet" or self.model=="Attention_R2UNet":
                self.attention = Attention_Block(F_g=out_chs, F_l=out_chs, F_int=out_chs//2)

            if self.model=="UNet" or self.model=="Attention_UNet":
                self.conv = Conv_Block(in_chs, out_chs)
            elif self.model=="R2UNet" or self.model=="Attention_R2UNet":
                self.conv = RRCNN_Block(in_chs, out_chs)


    def forward(self, x_old, x_new):
        x_new = self.up(x_new)
    
        if self.model=="Attention_UNet" or self.model=="Attention_R2UNet":
            x_old = self.attention(g=x_new, x=x_old)

        diffY = x_old.size()[2] - x_new.size()[2]
        diffX = x_old.size()[3] - x_new.size()[3]

        x_new = F.pad(x_new, [diffX // 2, diffX - diffX // 2, 
                              diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_old, x_new], dim=1)
        out = self.conv(x)
        return out


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False, model="UNet"):
        super(UNet, self).__init__()
        
        self.model = model
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.InConv = Conv_Block(n_channels, 64)

        self.Down1 = UNet_Down(64, 128, self.model)
        self.Down2 = UNet_Down(128, 256, self.model)
        self.Down3 = UNet_Down(256, 512, self.model)
        self.Down4 = UNet_Down(512, 1024 // factor,self.model)

        self.Up1 = UNet_Up(1024, 512 // factor, bilinear, self.model)
        self.Up2 = UNet_Up(512, 256 // factor, bilinear, self.model)
        self.Up3 = UNet_Up(256, 128 // factor, bilinear, self.model)
        self.Up4 = UNet_Up(128, 64, bilinear, self.model)

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
        print(out.size())
        return out
