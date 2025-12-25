import torch.nn as nn
from utils.modules import ConvBlock, DeconvBlock


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, nz=100, nc=3, ngf=64):
        super().__init__()

        channels = [ngf * 8, ngf * 4, ngf * 2, ngf]

        layers = []

        # first spatial layer
        layers.append(
            DeconvBlock(
                nz,
                channels[0],
                kernel_size=4,
                stride=1,
                padding=0,
                activation=nn.ReLU(inplace=True),
                norm="batch",
                bias=False,
                output_padding=0,
            )
        )

        # intermediate upsampling layers
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(
                DeconvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    activation=nn.ReLU(inplace=True),
                    norm="batch",
                    bias=False,
                    output_padding=0,
                )
            )

        # final image layer
        layers.append(
            DeconvBlock(
                channels[-1],
                nc,
                kernel_size=4,
                stride=2,
                padding=1,
                activation=nn.Tanh(),
                norm=None,
                bias=False,
                output_padding=0,
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()

        channels = [ndf, ndf * 2, ndf * 4, ndf * 8]

        layers = []

        # first layer (no bn)
        layers.append(
            ConvBlock(
                nc,
                channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                activation=nn.LeakyReLU(0.2, inplace=True),
                norm=None,
                bias=False,
            )
        )

        # intermediate layers
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    activation=nn.LeakyReLU(0.2, inplace=True),
                    norm="batch",
                    bias=False,
                )
            )

        # final classifier
        layers.append(nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0, norm="batch" ,bias=False))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out.view(out.size(0), -1).mean(dim=1)