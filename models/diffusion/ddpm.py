import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modules import SelfAttention


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DDPM(nn.Module):
    def __init__(
        self,
        img_ch=3,
        base_ch=128,
        ch_mults=(1, 2, 4, 4),
        attn_resolutions=(32, 16),
        time_dim=512,
        img_size=128
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.conv_in = nn.Conv2d(img_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = base_ch
        curr_res = img_size

        for mult in ch_mults:
            out_ch = base_ch * mult
            self.downs.append(
                nn.ModuleList([
                    ResBlock(ch, out_ch, time_dim),
                    SelfAttention(out_ch) if curr_res in attn_resolutions else nn.Identity(),
                    nn.Conv2d(out_ch, out_ch, 4, 2, 1) if curr_res > 4 else nn.Identity()
                ])
            )
            ch = out_ch
            curr_res //= 2

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_dim),
            SelfAttention(ch),
            ResBlock(ch, ch, time_dim)
        ])

        self.ups = nn.ModuleList()

        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            curr_res *= 2
            self.ups.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(ch, ch, 4, 2, 1),
                    ResBlock(ch + out_ch, out_ch, time_dim),
                    SelfAttention(out_ch) if curr_res in attn_resolutions else nn.Identity()
                ])
            )
            ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, img_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)

        x = self.conv_in(x)
        skips = []

        for res, attn, down in self.downs:
            x = res(x, t)
            x = attn(x)
            skips.append(x)
            x = down(x)

        x = self.mid[0](x, t)
        x = self.mid[1](x)
        x = self.mid[2](x, t)

        for up, res, attn in self.ups:
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = res(x, t)
            x = attn(x)

        return self.out(x)