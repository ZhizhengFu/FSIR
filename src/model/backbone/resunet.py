import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        res = self.conv(x)
        return res + x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False
        )
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.up(x)


class ResUNet(nn.Module):
    def __init__(
        self, in_channels=4, out_channels=3, channels=[64, 128, 256, 512], num_blocks=4
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(channels[0]),
            # nn.ReLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            *[ConvBlock(channels[0], channels[0]) for _ in range(num_blocks)],
            DownsampleBlock(channels[0], channels[1]),
        )
        self.down2 = nn.Sequential(
            *[ConvBlock(channels[1], channels[1]) for _ in range(num_blocks)],
            DownsampleBlock(channels[1], channels[2]),
        )
        self.down3 = nn.Sequential(
            *[ConvBlock(channels[2], channels[2]) for _ in range(num_blocks)],
            DownsampleBlock(channels[2], channels[3]),
        )

        self.body = nn.Sequential(
            *[ConvBlock(channels[3], channels[3]) for _ in range(num_blocks)]
        )

        self.up3 = nn.Sequential(
            UpsampleBlock(channels[3], channels[2]),
            *[ConvBlock(channels[2], channels[2]) for _ in range(num_blocks)],
        )
        self.up2 = nn.Sequential(
            UpsampleBlock(channels[2], channels[1]),
            *[ConvBlock(channels[1], channels[1]) for _ in range(num_blocks)],
        )
        self.up1 = nn.Sequential(
            UpsampleBlock(channels[1], channels[0]),
            *[ConvBlock(channels[0], channels[0]) for _ in range(num_blocks)],
        )

        self.tail = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (8 - w % 8) % 8, 0, (8 - h % 8) % 8), mode="replicate")

        x1 = self.head(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.body(x4)

        x = self.up3(x + x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.tail(x + x1)

        return x[..., :h, :w]
