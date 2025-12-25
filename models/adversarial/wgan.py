import torch.nn as nn
from utils.modules import ConvBlock, DeconvBlock


class Generator(nn.Module):
    def __init__(self, nz=100, nc=3, ngf=64):
        super().__init__()

        channels = [ngf * 8, ngf * 4, ngf * 2, ngf]

        # first linear layer
        self.l1 = nn.Sequential(
            nn.Linear(nz, channels[0] * 4 * 4),
            nn.BatchNorm1d(channels[0] * 4 * 4),
            nn.ReLU(inplace=True),
        )

        layers = []

        # intermediate upsampling layers
        for in_ch, out_ch in zip(channels[:-1], channels[0:]):
            layers.append(
                DeconvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    activation=nn.ReLU(inplace=True),
                    norm="batch",
                    bias=False,
                    output_padding=1,
                )
            )

        # final image layer
        layers.append(
            DeconvBlock(
                channels[-1],
                nc,
                kernel_size=5,
                stride=2,
                padding=2,
                activation=nn.Tanh(),
                norm=None,
                bias=False,
                output_padding=1,
            )
        )

        self.dblocks = nn.Sequential(*layers)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), -1, 4, 4)
        return self.dblocks(out)
    

class Critic(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()

        channels = [ndf, ndf * 2, ndf * 4, ndf * 8]
        layers = []

        layers.append(
            ConvBlock(
                nc,
                channels[0],
                kernel_size=5,
                stride=2,
                padding=2,
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
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    activation=nn.LeakyReLU(0.2, inplace=True),
                    norm="instance",
                    bias=False,
                )
            )

        # final classifier
        layers.append(nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1)