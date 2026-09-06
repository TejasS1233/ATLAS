import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates - main model for project
    Input: 4 channels (FLAIR, T1, T1ce, T2) stacked, Output: 1 channel mask
    src/model.py:35
    """
    def __init__(self, in_channels=4, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(features[3], features[3]*2)

        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], 2, stride=2)
        self.att4 = AttentionGate(features[3], features[3], features[3]//2)
        self.dec4 = DoubleConv(features[3]*2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.att3 = AttentionGate(features[2], features[2], features[2]//2)
        self.dec3 = DoubleConv(features[2]*2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.att2 = AttentionGate(features[1], features[1], features[1]//2)
        self.dec2 = DoubleConv(features[1]*2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.att1 = AttentionGate(features[0], features[0], features[0]//2)
        self.dec1 = DoubleConv(features[0]*2, features[0])

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        s4 = self.att4(d4, s4)
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        s3 = self.att3(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        s2 = self.att2(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        s1 = self.att1(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))

# Vanilla UNet for ablation comparison
class VanillaUNet(AttentionUNet):
    def forward(self, x):
        # same but without attention (for comparison)
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        s4 = self.enc4(p3)
        p4 = self.pool4(s4)
        b = self.bottleneck(p4)
        d4 = self.up4(b); d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return torch.sigmoid(self.final(d1))

def get_model(name="attention_unet"):
    if name == "attention_unet":
        return AttentionUNet(in_channels=4, out_channels=1)
    elif name == "vanilla":
        m = AttentionUNet(); m.__class__ = VanillaUNet; return m
    else:
        raise ValueError(name)

if __name__ == "__main__":
    model = get_model("attention_unet")
    x = torch.randn(2, 4, 128, 128)
    print(model(x).shape)  # should be [2,1,128,128]
