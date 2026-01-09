import torch
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


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=1, head_dim=None, norm="group", groups=32, dropout=0.0,
                 bias=True, use_residual=True, qkv_bias=None):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads
        self.inner_dim = self.head_dim * num_heads
        self.use_residual = use_residual

        assert (
            self.inner_dim == channels
        ), "channels must be divisible by num_heads * head_dim"

        # normalization
        if norm == "group":
            self.norm = nn.GroupNorm(groups, channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(channels)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {norm}")

        qkv_bias = bias if qkv_bias is None else qkv_bias

        # projections
        self.to_qkv = nn.Conv2d(channels, self.inner_dim * 3, 1, bias=qkv_bias)
        self.to_out = nn.Sequential(
            nn.Conv2d(self.inner_dim, channels, 1, bias=bias),
            nn.Dropout(dropout)
        )

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # reshape for multi-head
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 2, 3)

        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).reshape(b, self.inner_dim, h, w)

        out = self.to_out(out)

        return out + residual if self.use_residual else out
