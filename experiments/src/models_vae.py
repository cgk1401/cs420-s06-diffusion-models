"""
First-stage Perceptual Compression Autoencoder (VAE) for Latent Diffusion Models (LDM),
following Rombach et al. (2022):
  - Encoder E: x in R^{1 x 28 x 28} -> z in R^{4 x 7 x 7} (downsampling factor f = 4)
  - Decoder D: z in R^{4 x 7 x 7} -> x_tilde in R^{1 x 28 x 28}
  - Regularization: Mild KL-divergence penalty toward standard normal N(0, I).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialVAE(nn.Module):
    """
    2D Spatial Variational Autoencoder preserving spatial topology in latent space:
    Input:  [B, 1, 28, 28] (784 elements)
    Latent: [B, 4,  7,  7] (196 elements; spatial area reduced by f^2 = 16x)
    Output: [B, 1, 28, 28] in [-1.0, 1.0]
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 4, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder E: 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),      # [B, 32, 14, 14]
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1), # [B, 64, 7, 7]
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(base_channels * 2, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(base_channels * 2, latent_channels, kernel_size=1)

        # Decoder D: 7x7 -> 14x14 -> 28x28
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),  # [B, 32, 14, 14]
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1), # [B, 16, 28, 28]
            nn.SiLU(),
            nn.Conv2d(base_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1.0, 1.0]
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = torch.clamp(self.to_logvar(h), -10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        recon = self.decode(z)
        return recon, z, mu, logvar


def vae_loss_fn(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite reconstruction + mild KL-regularization loss (KL-reg LDM)."""
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
