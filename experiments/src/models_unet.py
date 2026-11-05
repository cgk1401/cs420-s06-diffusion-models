"""
Unified U-Net Noise Predictor with Sinusoidal Time Embedding and Cross-Attention Conditioning.
Designed so the exact same backbone architecture operates on:
  - Pixel Space (1 x 28 x 28) for DDPM
  - Latent Space (4 x 7 x 7) for LDM
enabling a strictly fair (apples-to-apples) computational and quality comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal timestep embedding for step t in {0, ..., T-1}."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class CrossAttentionConditionBlock(nn.Module):
    """
    Lightweight Cross-Attention conditioning block following Rombach et al. (2022):
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
    where Q is projected from spatial features phi(z_t) and K, V from condition embedding tau_theta(y).
    """

    def __init__(self, channels: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(4, channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(cond_dim, channels, bias=False)
        self.to_v = nn.Linear(cond_dim, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: spatial tensor [B, C, H, W]
        context: conditioning token sequence [B, L, cond_dim] (or [B, cond_dim])
        """
        b, c, h, w = x.shape
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [B, 1, cond_dim]

        h_in = self.norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]

        q = self.to_q(h_in).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, h * w, c)
        out = self.proj_out(out).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + out


class ResBlock(nn.Module):
    """Residual convolutional block modulated by timestep embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    """
    Compact U-Net noise predictor epsilon_theta(x_t, t, y).
    Works seamlessly on both 28x28 pixel tensors and 7x7 latent tensors.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_dim: int = 64,
        num_classes: int = 10,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.use_cross_attention = use_cross_attention

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Domain encoder tau_theta(y) for class/prompt conditioning
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)
        self.null_class_idx = num_classes

        c1 = base_channels
        c2 = base_channels * 2

        # +2 channels for normalized 2D spatial coordinates (u, v) in [-1, 1] for global spatial awareness
        self.in_conv = nn.Conv2d(in_channels + 2, c1, kernel_size=3, padding=1)

        # Encoder level 1
        self.down1 = ResBlock(c1, c1, time_dim)
        self.attn1 = CrossAttentionConditionBlock(c1, time_dim) if use_cross_attention else None

        # Bottleneck at same or pooled resolution (supports odd 7x7 and even 28x28 via interpolate)
        self.mid_down = ResBlock(c1, c2, time_dim)
        self.mid_attn = CrossAttentionConditionBlock(c2, time_dim) if use_cross_attention else None
        self.mid_up = ResBlock(c2, c1, time_dim)

        # Decoder level 1
        self.up1 = ResBlock(c1 * 2, c1, time_dim)
        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        t_emb = self.time_mlp(t)

        if y is None:
            y_idx = torch.full((b,), self.null_class_idx, device=x.device, dtype=torch.long)
        else:
            y_idx = y.long()
        cond_emb = self.class_emb(y_idx)  # tau_theta(y), shape [B, time_dim]

        # Additive conditioning + Cross-Attention conditioning
        t_cond = t_emb + cond_emb

        # Append normalized 2D spatial coordinate channels (u, v) in [-1, 1]
        gy = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        gx = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        x_in = torch.cat([x, gy, gx], dim=1)

        x0 = self.in_conv(x_in)
        d1 = self.down1(x0, t_cond)
        if self.attn1 is not None:
            d1 = self.attn1(d1, cond_emb)

        # Downsample spatially by 2x (works for both 28->14 and 7->4)
        low_h, low_w = max(2, h // 2), max(2, w // 2)
        b_in = F.avg_pool2d(d1, kernel_size=2, ceil_mode=True) if h >= 4 else d1
        b_mid = self.mid_down(b_in, t_cond)
        if self.mid_attn is not None:
            b_mid = self.mid_attn(b_mid, cond_emb)
        b_out = self.mid_up(b_mid, t_cond)
        b_up = F.interpolate(b_out, size=(h, w), mode="bilinear", align_corners=False)

        u1 = self.up1(torch.cat([b_up, d1], dim=1), t_cond)
        return self.out_conv(F.silu(self.out_norm(u1)))
