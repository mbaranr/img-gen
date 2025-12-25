import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 activation=None, norm="batch", bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.bn = nn.Identity()

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 activation=None, norm="batch", bias=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias, output_padding=output_padding
        )
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        self.activation = activation

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
