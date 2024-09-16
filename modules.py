import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, c_in, c_out, img_size):
        super(UNet, self).__init__()
        self.img_size = img_size

        self.down1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(64, c_out, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, t):
        # Downsampling
        d1 = self.down1(x)  # Shape: [B, 64, H, W]
        d2 = self.down2(F.max_pool2d(d1, 2))  # [B, 128, H/2, W/2]
        d3 = self.down3(F.max_pool2d(d2, 2))  # [B, 256, H/4, W/4]

        # Upsampling
        u1 = F.interpolate(d3, scale_factor=2, mode='nearest')  # [B, 256, H/2, W/2]
        u1 = torch.cat([u1, d2], dim=1)  # [B, 256+128, H/2, W/2]
        u1 = self.up_conv1(u1)  # [B, 128, H/2, W/2]

        u2 = F.interpolate(u1, scale_factor=2, mode='nearest')  # [B, 128, H, W]
        u2 = torch.cat([u2, d1], dim=1)  # [B, 128+64, H, W]
        u2 = self.up_conv2(u2)  # [B, 64, H, W]

        out = self.up_conv3(u2)  # [B, c_out, H, W]
        return out


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
