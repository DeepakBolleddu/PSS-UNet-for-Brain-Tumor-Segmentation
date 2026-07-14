"""
PSS-UNet (real selective-SSM version)
=====================================

This version makes the paper's central claim TRUE: it contains a genuine
selective state-space model (Mamba-style S6), not a pooled channel-gate.

Each Progressive State Module (PSM) now combines, as the paper describes:
  1. Squeeze-and-Excitation channel attention (Eqs. 1-3)
  2. A real selective SSM over the flattened spatial sequence (Eqs. 4-10):
       - state matrix A = -exp(A_log)
       - zero-order-hold discretization: Abar = exp(dt * A)
       - input-dependent B_t, C_t, dt_t (selectivity)
       - bidirectional scan (forward + reverse), since 3D images have no
         natural causal 1D order (this is the main accuracy-oriented upgrade)
  3. A global "progressive state" vector that flows across stages through the
     cross-scale bridges and produces a feature gate (Eqs. 11-13).

IMPORTANT, honest caveat about "matching the paper literally":
A real per-voxel selective scan over a 240^3 volume (8.8M positions) at EVERY
stage is not computationally tractable in plain PyTorch. The official paper
text claims SSM at every stage; in practice you can only afford it at the
lower-resolution stages. `ssm_stages` controls where the real scan runs.
Default = the low-resolution stages (enc3, enc4, bottleneck, dec4, dec3),
which is the honest, trainable interpretation. The highest-resolution stages
keep the cheap SE + global-state gating.

Drop-in: class name `PSSUNet`, same constructor (plus new optional kwargs with
safe defaults), same forward contract:
    training + deep_supervision -> (main_output, [aux_outputs], state_norm)
    otherwise                   -> main_output
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Optional fast path: the fused CUDA kernel from the `mamba-ssm` package.
# If installed, the selective scan runs in one kernel instead of a Python loop
# (orders of magnitude faster and far less memory). If not, we fall back to the
# correct pure-PyTorch sequential scan. Requires a CUDA build of mamba-ssm.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _MAMBA_SCAN
    _HAS_MAMBA = True
except Exception:
    _MAMBA_SCAN = None
    _HAS_MAMBA = False


# ============================================================================
# REAL SELECTIVE SSM (Mamba-style S6), 3D-aware via flattened spatial sequence
# ============================================================================

class SelectiveSSM3D(nn.Module):
    """
    Genuine selective state-space block.

    Operates on x: (B, C, D, H, W). Flattens spatial dims to a sequence of
    length L = D*H*W, runs a selective scan, reshapes back. Pre-norm residual
    with a SiLU gate (standard Mamba block layout). Bidirectional by default.

    This is intentionally a faithful, readable reference implementation. The
    scan is a sequential recurrence in PyTorch; it is correct but slower than
    the fused CUDA kernel from the `mamba-ssm` package. Use it at low-resolution
    stages, or swap in `selective_scan_fn` if you install mamba-ssm.
    """

    def __init__(self, d_model, d_state=16, expand=1, bidirectional=True,
                 use_checkpoint=True, use_fast='auto'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.bidirectional = bidirectional
        self.use_checkpoint = use_checkpoint
        # 'auto' uses the mamba-ssm kernel when available, else the slow scan.
        self.use_fast = _HAS_MAMBA if use_fast == 'auto' else bool(use_fast)

        self.norm = nn.LayerNorm(d_model)
        # in_proj -> [x_inner, gate]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        n_dir = 2 if bidirectional else 1
        # Per-direction selective-scan parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .repeat(n_dir, self.d_inner, 1))
        )  # (n_dir, d_inner, d_state)
        self.D = nn.Parameter(torch.ones(n_dir, self.d_inner))
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, d_state * 2, bias=False) for _ in range(n_dir)
        ])  # produces B_t and C_t
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.d_inner, bias=True) for _ in range(n_dir)
        ])

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Mamba-style dt initialization: spread initial timesteps over
        # [dt_min, dt_max] so the selective SSM learns useful timescales.
        # (A generic xavier+zero-bias init would make every channel start at
        # softplus(0)=0.69, which hurts the state model.)
        dt_min, dt_max = 1e-3, 0.1
        for dp in self.dt_proj:
            with torch.no_grad():
                dt = torch.exp(
                    torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                ).clamp(min=1e-4)
                dp.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def _scan_one_direction(self, x_seq, dir_idx):
        """x_seq: (B, L, d_inner) -> y: (B, L, d_inner)."""
        if self.use_fast:
            return self._scan_fast(x_seq, dir_idx)
        return self._scan_slow(x_seq, dir_idx)

    def _scan_fast(self, x_seq, dir_idx):
        """Fused mamba-ssm kernel. Equivalent math to _scan_slow (incl. D term)."""
        A = -torch.exp(self.A_log[dir_idx].float())          # (d_inner, d_state)
        D = self.D[dir_idx].float()                          # (d_inner,)
        bc = self.x_proj[dir_idx](x_seq)                     # (B, L, 2*d_state)
        B_t, C_t = bc.chunk(2, dim=-1)
        delta = self.dt_proj[dir_idx](x_seq)                 # (B, L, d_inner), raw (bias incl.)
        # mamba expects channel-first sequences: (B, d_inner|d_state, L)
        u = x_seq.transpose(1, 2).contiguous()               # (B, d_inner, L)
        delta = delta.transpose(1, 2).contiguous()           # (B, d_inner, L)
        Bm = B_t.transpose(1, 2).contiguous()                # (B, d_state, L)
        Cm = C_t.transpose(1, 2).contiguous()                # (B, d_state, L)
        # delta_softplus=True reproduces softplus(dt_proj(x)); bias already in delta.
        # NOTE: if your mamba-ssm version wants grouped B/C, use Bm.unsqueeze(1)/Cm.unsqueeze(1).
        y = _MAMBA_SCAN(u, delta, A, Bm, Cm, D=D, z=None,
                        delta_bias=None, delta_softplus=True)
        return y.transpose(1, 2)                             # (B, L, d_inner)

    def _scan_slow(self, x_seq, dir_idx):
        """Pure-PyTorch sequential reference scan (correct, slow)."""
        B, L, Dn = x_seq.shape
        A = -torch.exp(self.A_log[dir_idx].float())          # (d_inner, d_state)
        D = self.D[dir_idx].float()                          # (d_inner,)

        bc = self.x_proj[dir_idx](x_seq)                     # (B, L, 2*d_state)
        B_t, C_t = bc.chunk(2, dim=-1)                       # (B, L, d_state) each
        dt = F.softplus(self.dt_proj[dir_idx](x_seq))        # (B, L, d_inner)

        # Discretize (zero-order hold): dA = exp(dt * A)
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, Dn, self.d_state))
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(2)             # (B,L,d_inner,d_state)

        h = x_seq.new_zeros(B, Dn, self.d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_seq[:, t].unsqueeze(-1)
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(-1)       # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                           # (B, L, d_inner)
        y = y + x_seq * D.view(1, 1, Dn)
        return y

    def _forward_impl(self, x):
        B, C, D, H, W = x.shape
        L = D * H * W
        x_seq = x.flatten(2).transpose(1, 2)                 # (B, L, C)

        residual = x_seq
        x_seq = self.norm(x_seq)
        xz = self.in_proj(x_seq)                             # (B, L, 2*d_inner)
        x_in, gate = xz.chunk(2, dim=-1)
        x_in = F.silu(x_in)

        y = self._scan_one_direction(x_in, 0)
        if self.bidirectional:
            x_rev = torch.flip(x_in, dims=[1])
            y_rev = self._scan_one_direction(x_rev, 1)
            y = y + torch.flip(y_rev, dims=[1])

        y = y * F.silu(gate)
        y = self.out_proj(y)                                 # (B, L, C)
        y = y + residual
        return y.transpose(1, 2).reshape(B, C, D, H, W)

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ============================================================================
# SQUEEZE-AND-EXCITATION (real, Eqs. 1-3)
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        s = self.fc(x).view(b, c, 1, 1, 1)
        return x * s, s.view(b, c)


# ============================================================================
# PROGRESSIVE STATE MODULE (now genuinely SE + selective SSM + global state)
# ============================================================================

class ProgressiveStateModule(nn.Module):
    """
    Combines (per the paper):
      - SE channel attention                          -> X_tilde
      - (optional) real selective SSM over space      -> X_tilde + SSM(X_tilde)
      - global progressive state vector + feature gate (Eqs. 11-13)
    """

    def __init__(self, feature_dim, state_dim, prev_state_dim=None,
                 use_ssm=False, d_state=16, bidirectional=True,
                 use_checkpoint=True, use_residual=True, use_fast='auto'):
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.prev_state_dim = prev_state_dim if prev_state_dim is not None else state_dim
        self.use_residual = use_residual and (self.prev_state_dim == state_dim)
        self.use_ssm = use_ssm

        # SE
        self.se = SEBlock(feature_dim, reduction=16)

        # Real selective SSM (only built where requested)
        if use_ssm:
            self.ssm = SelectiveSSM3D(
                feature_dim, d_state=d_state, expand=1,
                bidirectional=bidirectional, use_checkpoint=use_checkpoint,
                use_fast=use_fast,
            )

        # Global progressive state
        self.state_proj = nn.Linear(self.prev_state_dim, state_dim, bias=False)
        self.feature_proj = nn.Linear(feature_dim, state_dim, bias=False)
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.state_activation = nn.Tanh()
        self.state_to_gate = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.initial_state = nn.Parameter(torch.zeros(1, state_dim))
        nn.init.xavier_uniform_(self.initial_state)
        if self.use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, prev_state=None):
        B, C, D, H, W = x.shape
        if prev_state is None:
            prev_state = self.initial_state.expand(B, -1)

        # 1) SE recalibration
        x_tilde, _ = self.se(x)

        # 2) Real selective SSM (spatial), residual
        if self.use_ssm:
            x_tilde = self.ssm(x_tilde)

        # 3) Global progressive state
        prev_state_proj = self.state_proj(prev_state)
        x_pooled = self.pool(x_tilde).view(B, C)
        x_proj = self.feature_proj(x_pooled)
        combined = torch.cat([prev_state_proj, x_proj], dim=1)
        new_state = self.state_activation(self.state_transition(combined))
        if self.use_residual:
            new_state = new_state + self.residual_gate * prev_state_proj

        # Feature gate, blended (paper Eq. 12: Y = g*X_tilde + (1-g)*X)
        gate = self.state_to_gate(new_state).view(B, C, 1, 1, 1)
        y = gate * x_tilde + (1.0 - gate) * x

        state_norm = torch.norm(new_state, dim=1).mean()
        return y, new_state, state_norm


# ============================================================================
# CROSS-SCALE STATE BRIDGE / STATE-AWARE SKIP (unchanged)
# ============================================================================

class CrossScaleStateBridge(nn.Module):
    def __init__(self, in_state_dim, out_state_dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_state_dim, out_state_dim),
            nn.LayerNorm(out_state_dim),
            nn.GELU(),
        )
        self.down_embed = nn.Parameter(torch.randn(1, out_state_dim) * 0.02)
        self.up_embed = nn.Parameter(torch.randn(1, out_state_dim) * 0.02)

    def forward(self, state, direction='down'):
        out = self.transform(state)
        return out + (self.down_embed if direction == 'down' else self.up_embed)


class StateAwareSkip(nn.Module):
    def __init__(self, feature_dim, enc_state_dim, dec_state_dim):
        super().__init__()
        self.state_fusion = nn.Sequential(
            nn.Linear(enc_state_dim + dec_state_dim, dec_state_dim),
            nn.LayerNorm(dec_state_dim),
            nn.GELU(),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(dec_state_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, dec_state, enc_state, enc_features):
        fused_state = self.state_fusion(torch.cat([dec_state, enc_state], dim=1))
        gate = self.feature_gate(fused_state)
        b, c = gate.shape
        return enc_features * gate.view(b, c, 1, 1, 1), fused_state


# ============================================================================
# CONV BLOCKS (now residual, a cheap V-Net-style upgrade)
# ============================================================================

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv3d(in_ch, out_ch, 1, bias=False))
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = ResConvBlock(out_ch, out_ch)

    def forward(self, x):
        return self.res(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.res = ResConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))


class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels // 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels // 2, out_channels, 1),
        )

    def forward(self, x, target_size):
        out = self.conv(x)
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='trilinear', align_corners=False)
        return out


# ============================================================================
# PSS-UNET (real-SSM, drop-in)
# ============================================================================

class PSSUNet(nn.Module):
    # The pure-PyTorch sequential scan is only fast enough at the smallest
    # stages (enc4/bottleneck, L=15^3~3.4k at 240^3 input). dec4 (L~17k) makes
    # the first epoch take ~1h, so it is dropped by default. To add dec4/enc3/
    # dec3, install mamba-ssm (the fused kernel makes it cheap) and they will be
    # used automatically.
    # Pure-PyTorch scan is only fast at the smallest stages (enc4/bottleneck,
    # L~3.4k at 240^3). With mamba-ssm installed, the fused kernel makes the
    # wider 'SSM at every stage' set cheap, so we widen automatically.
    NARROW_SSM_STAGES = ('enc4', 'bottleneck')
    WIDE_SSM_STAGES   = ('enc3', 'enc4', 'bottleneck', 'dec4', 'dec3')
    DEFAULT_SSM_STAGES = WIDE_SSM_STAGES if _HAS_MAMBA else NARROW_SSM_STAGES

    def __init__(self, in_channels=4, out_channels=1, base_filters=24,
                 state_dim=64, use_checkpoint=True, deep_supervision=True,
                 ssm_stages=None, d_state=16, bidirectional=True, use_fast='auto'):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.deep_supervision = deep_supervision
        ssm_stages = set(self.DEFAULT_SSM_STAGES if ssm_stages is None else ssm_stages)
        self.ssm_stages = ssm_stages

        f, s = base_filters, state_dim

        def psm(feature_dim, sd, prev, stage):
            return ProgressiveStateModule(
                feature_dim, sd, prev_state_dim=prev,
                use_ssm=(stage in ssm_stages), d_state=d_state,
                bidirectional=bidirectional, use_checkpoint=use_checkpoint,
                use_fast=use_fast,
            )

        # Encoder
        self.input_conv = ResConvBlock(in_channels, f)
        self.input_state = psm(f, s, s, 'input')

        self.enc1 = DownBlock(f, f * 2);   self.bridge1 = CrossScaleStateBridge(s, s)
        self.enc1_state = psm(f * 2, s, s, 'enc1')
        self.enc2 = DownBlock(f * 2, f * 4); self.bridge2 = CrossScaleStateBridge(s, s)
        self.enc2_state = psm(f * 4, s, s, 'enc2')
        self.enc3 = DownBlock(f * 4, f * 8); self.bridge3 = CrossScaleStateBridge(s, s)
        self.enc3_state = psm(f * 8, s, s, 'enc3')
        self.enc4 = DownBlock(f * 8, f * 8); self.bridge4 = CrossScaleStateBridge(s, s * 2)
        self.enc4_state = psm(f * 8, s * 2, s * 2, 'enc4')

        # Bottleneck
        self.bottleneck = ResConvBlock(f * 8, f * 8)
        self.bottleneck_state = psm(f * 8, s * 2, s * 2, 'bottleneck')

        # Decoder
        self.up_bridge4 = CrossScaleStateBridge(s * 2, s)
        self.skip4 = StateAwareSkip(f * 8, s, s)
        self.dec4 = UpBlock(f * 8, f * 8, f * 8); self.dec4_state = psm(f * 8, s, s, 'dec4')

        self.up_bridge3 = CrossScaleStateBridge(s, s)
        self.skip3 = StateAwareSkip(f * 4, s, s)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4); self.dec3_state = psm(f * 4, s, s, 'dec3')

        self.up_bridge2 = CrossScaleStateBridge(s, s)
        self.skip2 = StateAwareSkip(f * 2, s, s)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2); self.dec2_state = psm(f * 2, s, s, 'dec2')

        self.up_bridge1 = CrossScaleStateBridge(s, s)
        self.skip1 = StateAwareSkip(f, s, s)
        self.dec1 = UpBlock(f * 2, f, f); self.dec1_state = psm(f, s, s, 'dec1')

        # Output
        self.output = nn.Conv3d(f, out_channels, 1)
        if deep_supervision:
            self.ds_head4 = DeepSupervisionHead(f * 8, out_channels)
            self.ds_head3 = DeepSupervisionHead(f * 4, out_channels)
            self.ds_head2 = DeepSupervisionHead(f * 2, out_channels)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if '.ssm.' in name or name.endswith('ssm'):
                continue  # SelectiveSSM3D keeps its own Mamba-style init
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        target_size = x.shape[2:]
        norms = []

        x0 = self._ckpt(self.input_conv, x)
        x0, st0, n = self.input_state(x0, None); norms.append(n)

        e1 = self._ckpt(self.enc1, x0)
        e1, st1, n = self.enc1_state(e1, self.bridge1(st0, 'down')); norms.append(n)
        e2 = self._ckpt(self.enc2, e1)
        e2, st2, n = self.enc2_state(e2, self.bridge2(st1, 'down')); norms.append(n)
        e3 = self._ckpt(self.enc3, e2)
        e3, st3, n = self.enc3_state(e3, self.bridge3(st2, 'down')); norms.append(n)
        e4 = self._ckpt(self.enc4, e3)
        e4, st4, n = self.enc4_state(e4, self.bridge4(st3, 'down')); norms.append(n)

        bn = self._ckpt(self.bottleneck, e4)
        bn, st_bn, n = self.bottleneck_state(bn, st4); norms.append(n)

        aux = []
        f4, ds4 = self.skip4(self.up_bridge4(st_bn, 'up'), st3, e3)
        d4 = self.dec4(bn, f4)
        d4, ds4, n = self.dec4_state(d4, ds4); norms.append(n)
        if self.deep_supervision and self.training:
            aux.append(self.ds_head4(d4, target_size))

        f3, ds3 = self.skip3(self.up_bridge3(ds4, 'up'), st2, e2)
        d3 = self.dec3(d4, f3)
        d3, ds3, n = self.dec3_state(d3, ds3); norms.append(n)
        if self.deep_supervision and self.training:
            aux.append(self.ds_head3(d3, target_size))

        f2, ds2 = self.skip2(self.up_bridge2(ds3, 'up'), st1, e1)
        d2 = self.dec2(d3, f2)
        d2, ds2, n = self.dec2_state(d2, ds2); norms.append(n)
        if self.deep_supervision and self.training:
            aux.append(self.ds_head2(d2, target_size))

        f1, ds1 = self.skip1(self.up_bridge1(ds2, 'up'), st0, x0)
        d1 = self.dec1(d2, f1)
        d1, _, n = self.dec1_state(d1, ds1); norms.append(n)

        main = self.output(d1)
        total_norm = torch.stack(norms).mean()
        if self.training and self.deep_supervision:
            return main, aux, total_norm
        return main


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Small smoke test: keep SSM at low-res stages so the sequence stays short.
    m = PSSUNet(base_filters=8, state_dim=32, use_checkpoint=False,
                ssm_stages=('enc3', 'enc4', 'bottleneck', 'dec4'),
                d_state=8).to(dev)
    print(f"Params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    x = torch.randn(1, 4, 48, 48, 48, device=dev)
    m.train()
    out, aux, sn = m(x)
    print("train:", out.shape, [a.shape for a in aux], float(sn))
    m.eval()
    with torch.no_grad():
        print("eval :", m(x).shape)
    print("OK")