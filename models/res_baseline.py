#!/usr/bin/env python3
"""
res_baseline.py  --  E0 control for the revision.

PSS-UNet's backbone with EVERY progressive-state component removed:
no ProgressiveStateModule (no SE, no selective scan, no state gate), no
cross-scale bridges, no state-aware skips. Skips are plain concatenation.
Everything else is byte-for-byte the PSS-UNet backbone: the same
ResConvBlock / DownBlock / UpBlock classes imported from pss_unet.py, the same
widths (24, 48, 96, 192, 192), the same strided-conv down / trilinear up, the
same weight initialisation, the same gradient checkpointing during training.

Parameter count: 12.67M (the number stated in Section 3.3 / Appendix A of the paper).

Train / evaluate exactly like the other models:
    python train_fair.py --model res_baseline --seed 42 ... --deep_sup none
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pss_unet import ResConvBlock, DownBlock, UpBlock   # the identical backbone blocks


class ResBaseline(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=24, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        f = base_filters

        # Encoder (same layer names and shapes as PSSUNet, minus the *_state modules)
        self.input_conv = ResConvBlock(in_channels, f)
        self.enc1 = DownBlock(f, f * 2)
        self.enc2 = DownBlock(f * 2, f * 4)
        self.enc3 = DownBlock(f * 4, f * 8)
        self.enc4 = DownBlock(f * 8, f * 8)

        # Bottleneck
        self.bottleneck = ResConvBlock(f * 8, f * 8)

        # Decoder (plain concatenation skips: UpBlock concatenates [up(x), skip])
        self.dec4 = UpBlock(f * 8, f * 8, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpBlock(f * 2, f, f)

        # Output
        self.output = nn.Conv3d(f, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        # identical to PSSUNet._init_weights for the layers that exist here
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.01)

    def _ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        x0 = self._ckpt(self.input_conv, x)
        e1 = self._ckpt(self.enc1, x0)
        e2 = self._ckpt(self.enc2, e1)
        e3 = self._ckpt(self.enc3, e2)
        e4 = self._ckpt(self.enc4, e3)
        bn = self._ckpt(self.bottleneck, e4)
        d4 = self.dec4(bn, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)
        return self.output(d1)          # single logit volume, like the other controls


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ResBaseline().to(dev)
    n = sum(p.numel() for p in m.parameters())
    print(f"Params: {n:,} ({n/1e6:.2f}M)  -- expected 12.67M")
    assert abs(n / 1e6 - 12.67) < 0.01, "parameter count does not match the paper"
    x = torch.randn(1, 4, 64, 64, 64, device=dev)
    m.eval()
    with torch.no_grad():
        print("out:", m(x).shape)
    print("OK")