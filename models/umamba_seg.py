#!/usr/bin/env python3
"""
umamba_seg.py

A U-Mamba-style baseline: a standard 3D U-Net with a GENUINE selective-scan
Mamba block at the bottleneck, using the fused `mamba_ssm` kernel (the same
Mamba layer verified to run on the project's V100). This is a real state-space
model, unlike the placeholder scan in minimal_mamba_vnet.py.

Drop-in for train_fair.py build_model:
    from umamba_seg import UMambaSeg
    return UMambaSeg(in_channels=4, out_channels=1)

Same forward contract as the other controlled-study models: returns a single
logit volume (no deep supervision), so it trains under --deep_sup none exactly
like baseline / vnet_se / vnet_ssm.

Requires mamba-ssm (fused kernel) available at runtime; load the CUDA module
(module load cuda/12.2.2) in the job so the kernel is present.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba  # verified working on V100 with causal_conv1d 1.4.0


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class MambaBottleneck(nn.Module):
    """Genuine selective-scan Mamba over the flattened bottleneck sequence.
    Pre-norm residual, bidirectional (forward + reversed) since a 3D volume has
    no natural causal order."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        b, c, d, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)          # (B, L, C)
        res = seq
        seq = self.norm(seq)
        y = self.mamba_f(seq)
        y = y + torch.flip(self.mamba_b(torch.flip(seq, dims=[1])), dims=[1])
        y = y + res
        return y.transpose(1, 2).reshape(b, c, d, h, w)


class UMambaSeg(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, n_filters=24,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        f = n_filters
        self.in_conv = ConvBlock(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)

        self.bottleneck_conv = ConvBlock(f * 8, f * 8)
        self.mamba = MambaBottleneck(f * 8, d_state=d_state, d_conv=d_conv, expand=expand)

        self.up3 = Up(f * 8, f * 8, f * 4)
        self.up2 = Up(f * 4, f * 4, f * 2)
        self.up1 = Up(f * 2, f * 2, f)
        self.out = nn.Conv3d(f, out_channels, 1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1, s1 = self.down1(x0)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        bn = self.bottleneck_conv(x3)
        bn = self.mamba(bn)
        d3 = self.up3(bn, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        # align to input size, then final skip with x0
        if d1.shape[2:] != x0.shape[2:]:
            d1 = F.interpolate(d1, size=x0.shape[2:], mode="trilinear", align_corners=False)
        return self.out(d1)


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = UMambaSeg(in_channels=4, out_channels=1, n_filters=16).to(dev)
    print(f"Params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    x = torch.randn(1, 4, 64, 64, 64, device=dev)
    print("out:", m(x).shape)