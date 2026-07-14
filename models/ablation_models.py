"""
Ablation Study Models for Pro-SSUNet

This file contains all model variants for proper ablation study:
- E1: BaselineVNet (no SE, no SSM)
- E2: VNetWithSE (SE attention only, no SSM)  
- E3: VNetWithSSM (SSM only, no SE)
- E4: ProSSUNet (SE + SSM) - Already trained, just reference

CRITICAL: All models use the SAME:
- Base architecture (encoder-decoder structure)
- Number of filters (base_filters=24)
- Normalization (InstanceNorm3d)
- Activation (LeakyReLU 0.01)
- Input/output dimensions

This ensures fair comparison for ablation study.

Author: Deepak
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


# ============================================================================
# SHARED BUILDING BLOCKS (Same for all models)
# ============================================================================

class ConvBlock(nn.Module):
    """Standard 3D Conv block - shared across all ablation models"""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv3d(in_ch, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ])
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block with strided convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with trilinear interpolation"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
    
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ============================================================================
# CHANNEL ATTENTION (SE Block)
# ============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.shape[:2]
        w = self.fc(x).view(b, c, 1, 1, 1)
        return x * w


# ============================================================================
# LIGHTWEIGHT SSM (State Space Model)
# ============================================================================

class LightweightSSM(nn.Module):
    """
    Ultra memory-efficient SSM for 3D medical imaging.
    Same implementation as in Pro-SSUNet.
    """
    def __init__(self, d_model, d_state=4, chunk_size=512):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # BC and dt projections
        self.bc_proj = nn.Linear(d_model, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        L = D * H * W
        
        # Flatten spatial dimensions
        x_flat = x.flatten(2).transpose(1, 2)  # (B, L, C)
        
        # Project
        xz = self.in_proj(x_flat)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # SSM forward
        y = self._ssm_forward(x_ssm)
        
        # Gate and project
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        
        # Residual
        y = y + x_flat
        
        return y.transpose(1, 2).reshape(B, C, D, H, W)
    
    def _ssm_forward(self, x):
        """Chunked SSM computation"""
        B, L, D = x.shape
        
        A = -torch.exp(self.A_log.float())
        
        bc = self.bc_proj(x)
        B_ssm, C_ssm = bc.chunk(2, dim=-1)
        delta = F.softplus(self.dt_proj(x))
        
        # Process in chunks for memory efficiency
        outputs = []
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            
            x_chunk = x[:, start:end]
            delta_chunk = delta[:, start:end]
            B_chunk = B_ssm[:, start:end]
            C_chunk = C_ssm[:, start:end]
            
            chunk_out = []
            for t in range(end - start):
                dA = torch.exp(delta_chunk[:, t:t+1, :, None] * A[None, None, :, :])
                dB = delta_chunk[:, t, :, None] * B_chunk[:, t, None, :]
                
                h = dA.squeeze(1) * h + dB * x_chunk[:, t, :, None]
                y_t = (h * C_chunk[:, t, None, :]).sum(-1)
                chunk_out.append(y_t)
            
            outputs.append(torch.stack(chunk_out, dim=1))
        
        y = torch.cat(outputs, dim=1)
        y = y + x * self.D[None, None, :]
        
        return y


class SSMBlock(nn.Module):
    """SSM block with layer normalization"""
    def __init__(self, d_model, d_state=4, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(d_model)
        self.ssm = LightweightSSM(d_model, d_state)
    
    def _forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat).transpose(1, 2).reshape(B, C, D, H, W)
        return self.ssm(x_norm)
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# ============================================================================
# E1: BASELINE V-NET (No SE, No SSM)
# ============================================================================

class BaselineVNet(nn.Module):
    """
    E1: Baseline V-Net without any enhancements.
    
    This is the control model for ablation study.
    Same architecture as Pro-SSUNet but without SE attention and SSM.
    """
    def __init__(self, in_channels=4, out_channels=1, base_filters=24, use_checkpoint=True, **kwargs):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        f = base_filters
        
        # Input processing (same as Pro-SSUNet)
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f, f, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # Encoder (same structure as Pro-SSUNet)
        self.enc1 = DownBlock(f, f * 2)
        self.enc2 = DownBlock(f * 2, f * 4)
        self.enc3 = DownBlock(f * 4, f * 8)
        self.enc4 = DownBlock(f * 8, f * 8)
        
        # Bottleneck (standard conv, NO SSM)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(f * 8, f * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f * 8, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f * 8, f * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f * 8, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # Decoder
        self.dec4 = UpBlock(f * 8, f * 8, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpBlock(f * 2, f, f)
        
        # Output
        self.output = nn.Conv3d(f, out_channels, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)
    
    def forward(self, x):
        # Input
        x0 = self._ckpt(self.input_conv, x)
        
        # Encoder (save skip connections)
        x1 = self._ckpt(self.enc1, x0)
        x2 = self._ckpt(self.enc2, x1)
        x3 = self._ckpt(self.enc3, x2)
        x4 = self._ckpt(self.enc4, x3)
        
        # Bottleneck (NO SSM)
        bn = self._ckpt(self.bottleneck, x4)
        
        # Decoder
        d4 = self.dec4(bn, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        return self.output(d1)


# ============================================================================
# E2: V-NET WITH SE ATTENTION (SE only, No SSM)
# ============================================================================

class VNetWithSE(nn.Module):
    """
    E2: V-Net with SE attention but without SSM.
    
    Tests the contribution of channel attention alone.
    """
    def __init__(self, in_channels=4, out_channels=1, base_filters=24, use_checkpoint=True, **kwargs):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        f = base_filters
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f, f, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # Encoder with SE attention after each level
        self.enc1 = DownBlock(f, f * 2)
        self.se1 = ChannelAttention(f * 2)
        
        self.enc2 = DownBlock(f * 2, f * 4)
        self.se2 = ChannelAttention(f * 4)
        
        self.enc3 = DownBlock(f * 4, f * 8)
        self.se3 = ChannelAttention(f * 8)
        
        self.enc4 = DownBlock(f * 8, f * 8)
        self.se4 = ChannelAttention(f * 8)
        
        # Bottleneck (NO SSM, just conv)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(f * 8, f * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f * 8, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f * 8, f * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f * 8, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # Decoder
        self.dec4 = UpBlock(f * 8, f * 8, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpBlock(f * 2, f, f)
        
        # Output
        self.output = nn.Conv3d(f, out_channels, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)
    
    def forward(self, x):
        # Input
        x0 = self._ckpt(self.input_conv, x)
        
        # Encoder with SE attention
        x1 = self.se1(self._ckpt(self.enc1, x0))
        x2 = self.se2(self._ckpt(self.enc2, x1))
        x3 = self.se3(self._ckpt(self.enc3, x2))
        x4 = self.se4(self._ckpt(self.enc4, x3))
        
        # Bottleneck
        bn = self._ckpt(self.bottleneck, x4)
        
        # Decoder
        d4 = self.dec4(bn, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        return self.output(d1)


# ============================================================================
# E3: V-NET WITH SSM (SSM only, No SE)
# ============================================================================

class VNetWithSSM(nn.Module):
    """
    E3: V-Net with SSM bottleneck but without SE attention.
    
    Tests the contribution of state space model alone.
    """
    def __init__(self, in_channels=4, out_channels=1, base_filters=24, d_state=4, use_checkpoint=True, **kwargs):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        f = base_filters
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f, f, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # Encoder (NO SE attention)
        self.enc1 = DownBlock(f, f * 2)
        self.enc2 = DownBlock(f * 2, f * 4)
        self.enc3 = DownBlock(f * 4, f * 8)
        self.enc4 = DownBlock(f * 8, f * 8)
        
        # Bottleneck WITH SSM
        self.bottleneck = nn.Sequential(
            nn.Conv3d(f * 8, f * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f * 8, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.ssm = SSMBlock(f * 8, d_state=d_state, use_checkpoint=use_checkpoint)
        
        # Decoder
        self.dec4 = UpBlock(f * 8, f * 8, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpBlock(f * 2, f, f)
        
        # Output
        self.output = nn.Conv3d(f, out_channels, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)
    
    def forward(self, x):
        # Input
        x0 = self._ckpt(self.input_conv, x)
        
        # Encoder (NO SE)
        x1 = self._ckpt(self.enc1, x0)
        x2 = self._ckpt(self.enc2, x1)
        x3 = self._ckpt(self.enc3, x2)
        x4 = self._ckpt(self.enc4, x3)
        
        # Bottleneck with SSM
        bn = self._ckpt(self.bottleneck, x4)
        bn = self.ssm(bn)
        
        # Decoder
        d4 = self.dec4(bn, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        return self.output(d1)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def get_ablation_model(model_name, **kwargs):
    """
    Factory function to get ablation model by name.
    
    Args:
        model_name: One of ['baseline', 'vnet_se', 'vnet_ssm', 'prossunet']
        
    Returns:
        Model instance
    """
    models = {
        'baseline': BaselineVNet,
        'vnet_se': VNetWithSE,
        'vnet_ssm': VNetWithSSM,
    }
    
    if model_name == 'prossunet':
        # Import the actual Pro-SSUNet from your trained model
        from pro_ssunet_corrected import ProSSUNet
        return ProSSUNet(**kwargs)
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys()) + ['prossunet']}")
    
    return models[model_name](**kwargs)


# ============================================================================
# PARAMETER COUNT COMPARISON
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_comparison():
    """Print parameter count for all ablation models"""
    print("\n" + "=" * 60)
    print("ABLATION MODEL PARAMETER COMPARISON")
    print("=" * 60)
    
    configs = {'base_filters': 24, 'd_state': 4, 'use_checkpoint': True}
    
    for name in ['baseline', 'vnet_se', 'vnet_ssm']:
        model = get_ablation_model(name, **configs)
        params = count_parameters(model)
        print(f"{name:15s}: {params:,} ({params/1e6:.2f}M)")
    
    # Pro-SSUNet
    try:
        from pro_ssunet_corrected import ProSSUNet
        model = ProSSUNet(**configs)
        params = count_parameters(model)
        print(f"{'prossunet':15s}: {params:,} ({params/1e6:.2f}M)")
    except ImportError:
        print("prossunet: (import pro_ssunet_new to see)")
    
    print("=" * 60)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing Ablation Models...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test input
    x = torch.randn(1, 4, 128, 128, 128).to(device)
    
    configs = {'base_filters': 24, 'd_state': 4, 'use_checkpoint': True}
    
    for name in ['baseline', 'vnet_se', 'vnet_ssm']:
        print(f"\nTesting {name}...")
        model = get_ablation_model(name, **configs).to(device)
        params = count_parameters(model)
        
        with torch.no_grad():
            y = model(x)
        
        print(f"  Parameters: {params:,}")
        print(f"  Input: {x.shape}, Output: {y.shape}")
    
    print_model_comparison()
    print("\nAll tests passed!")
