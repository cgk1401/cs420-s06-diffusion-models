"""
Core mathematical formulations for Denoising Diffusion Probabilistic Models (DDPM)
and Latent Diffusion Models (LDM), following Ho et al. (2020) and Rombach et al. (2022).

Equations implemented:
1. Step-wise forward transition:
   q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
2. Closed-form arbitrary-step forward sampling:
   q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,  epsilon ~ N(0, I)
3. Reverse denoising transition:
   x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
4. Simplified training objective:
   L_simple = E_{t, x_0, epsilon}[ || epsilon - epsilon_theta(x_t, t) ||^2 ]
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DiffusionSchedule:
    """Precomputed DDPM variance schedule and closed-form coefficients."""
    T: int
    betas: torch.Tensor                  # beta_t in (0, 1), shape [T]
    alphas: torch.Tensor                 # alpha_t = 1 - beta_t, shape [T]
    alphas_cumprod: torch.Tensor         # alpha_bar_t = prod_{s=1}^t alpha_s, shape [T]
    alphas_cumprod_prev: torch.Tensor    # alpha_bar_{t-1} (with alpha_bar_0 = 1.0), shape [T]
    sqrt_alphas: torch.Tensor            # sqrt(alpha_t)
    sqrt_one_minus_betas: torch.Tensor   # sqrt(1 - beta_t)
    sqrt_betas: torch.Tensor             # sqrt(beta_t)
    sqrt_alphas_cumprod: torch.Tensor    # sqrt(alpha_bar_t)
    sqrt_one_minus_alphas_cumprod: torch.Tensor  # sqrt(1 - alpha_bar_t)
    posterior_variance: torch.Tensor     # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)


def build_linear_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: str | torch.device = "cpu",
) -> DiffusionSchedule:
    """Create the standard linear beta schedule from Ho et al. (2020)."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=torch.float32, device=device), alphas_cumprod[:-1]], dim=0
    )

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return DiffusionSchedule(
        T=T,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas=torch.sqrt(alphas),
        sqrt_one_minus_betas=torch.sqrt(1.0 - betas),
        sqrt_betas=torch.sqrt(betas),
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
        posterior_variance=posterior_variance,
    )


def _extract(coeff: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract coefficient at timestep tensor `t` and reshape to broadcast over `x_shape`."""
    t = t.to(coeff.device)
    batch_size = t.shape[0]
    out = coeff.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample_one_step(
    x_prev: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single Markov step forward transition:
    q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
    """
    if noise is None:
        noise = torch.randn_like(x_prev)
    mean_coeff = _extract(schedule.sqrt_one_minus_betas, t, x_prev.shape)
    std_coeff = _extract(schedule.sqrt_betas, t, x_prev.shape)
    x_t = mean_coeff * x_prev + std_coeff * noise
    return x_t, noise


def q_sample_closed_form(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form forward noising from clean data x0 directly to arbitrary step t:
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = _extract(schedule.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alpha_bar = _extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise


def compute_l_simple(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    y: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute simplified noise-prediction objective:
    L_simple = E[ || epsilon - epsilon_theta(x_t, t, y) ||^2 ]
    Returns (loss, true_noise, predicted_noise).
    """
    x_t, eps = q_sample_closed_form(x0, t, schedule, noise=noise)
    eps_pred = model(x_t, t, y)
    loss = F.mse_loss(eps_pred, eps)
    return loss, eps, eps_pred


@torch.no_grad()
def p_sample_step(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t_index: int,
    schedule: DiffusionSchedule,
    y: torch.Tensor | None = None,
    z_noise: torch.Tensor | None = None,
    clip_x0: bool = True,
    clip_range: float = 1.0,
) -> torch.Tensor:
    """
    Single reverse denoising step from t_index to t_index - 1 using Ho et al. (2020) Eq. (7) & (11):
    1. Predict x_0: x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
    2. Compute posterior mean mu_tilde_t(x_t, x0_hat) =
       (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x0_hat
       + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
    """
    b = x_t.shape[0]
    t_tensor = torch.full((b,), t_index, device=x_t.device, dtype=torch.long)
    eps_theta = model(x_t, t_tensor, y)

    beta_t = schedule.betas[t_index]
    alpha_t = schedule.alphas[t_index]
    sqrt_alpha_t = schedule.sqrt_alphas[t_index]
    alpha_bar_t = schedule.alphas_cumprod[t_index]
    alpha_bar_prev = schedule.alphas_cumprod_prev[t_index]
    sqrt_alpha_bar_t = schedule.sqrt_alphas_cumprod[t_index]
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alphas_cumprod[t_index]

    x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
    if clip_x0:
        x0_hat = torch.clamp(x0_hat, -clip_range, clip_range)

    coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coef2 = sqrt_alpha_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    mean = coef1 * x0_hat + coef2 * x_t

    if t_index == 0:
        return x0_hat

    sigma_t = torch.sqrt(schedule.posterior_variance[t_index])
    if z_noise is None:
        z_noise = torch.randn_like(x_t)
    return mean + sigma_t * z_noise


@torch.no_grad()
def p_sample_loop(
    model: torch.nn.Module,
    shape: tuple[int, ...],
    schedule: DiffusionSchedule,
    y: torch.Tensor | None = None,
    steps_to_record: list[int] | None = None,
    x_T: torch.Tensor | None = None,
    seed: int | None = None,
    clip_x0: bool = True,
    clip_range: float = 1.0,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Full sequential reverse denoising chain from t = T-1 down to t = 0.
    Cannot skip steps in standard DDPM.
    """
    device = schedule.betas.device
    if seed is not None:
        torch.manual_seed(seed)

    x_t = torch.randn(shape, device=device) if x_T is None else x_T.clone()
    recorded: dict[int, torch.Tensor] = {}
    record_set = set(steps_to_record or [])

    for t_idx in range(schedule.T - 1, -1, -1):
        if t_idx in record_set and t_idx == schedule.T - 1:
            recorded[t_idx] = x_t.detach().cpu().clone()
        x_t = p_sample_step(
            model, x_t, t_idx, schedule, y=y, clip_x0=clip_x0, clip_range=clip_range
        )
        if t_idx in record_set and t_idx != schedule.T - 1:
            recorded[t_idx] = x_t.detach().cpu().clone()

    return x_t, recorded
