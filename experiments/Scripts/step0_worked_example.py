"""
Step 0: 1D Worked Numerical Example Generator & Verifier (Section 3.2 — Task 3 & Section III).
Extracts a real foreground pixel from the canonical 28x28 MNIST digit '3' image and
uses the exact same random seed (SEED=42) and Gaussian noise tensor used in Section 4.
"""

import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset import get_canonical_digit_sample
from src.diffusion import build_linear_schedule, q_sample_one_step, q_sample_closed_form
from src.models_unet import ConditionalUNet


def main():
    torch.manual_seed(config.SEED)
    schedule = build_linear_schedule(
        T=config.T,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device="cpu",
    )

    # 1. Load canonical 28x28 digit '3' and its foreground pixel (r, c)
    img_x0, label, (r, c) = get_canonical_digit_sample(digit=3)
    x0_batch = img_x0.unsqueeze(0)  # [1, 1, 28, 28]

    # 2. Sample the canonical Gaussian noise tensor with SEED=42
    torch.manual_seed(config.SEED)
    eps_tensor = torch.randn_like(x0_batch)

    # Extract 1D scalar values at pixel (r, c)
    x0_val = float(x0_batch[0, 0, r, c].item())
    raw_pixel_255 = round((x0_val + 1.0) * 127.5)
    eps_val = float(eps_tensor[0, 0, r, c].item())

    # --- Part A1: Step t = 1 with exact DDPM schedule beta_1 = 0.0001 ---
    t1_idx = 0  # 0-based index for t = 1
    beta_1 = float(schedule.betas[t1_idx].item())
    alpha_1 = float(schedule.alphas[t1_idx].item())
    sqrt_one_minus_beta1 = float(schedule.sqrt_one_minus_betas[t1_idx].item())
    sqrt_beta1 = float(schedule.sqrt_betas[t1_idx].item())

    cond_mean_t1 = sqrt_one_minus_beta1 * x0_val
    noise_term_t1 = sqrt_beta1 * eps_val
    x1_val = cond_mean_t1 + noise_term_t1

    # Verify against tensor function q_sample_one_step
    x1_tensor, _ = q_sample_one_step(
        x0_batch, torch.tensor([t1_idx], dtype=torch.long), schedule, noise=eps_tensor
    )
    assert abs(float(x1_tensor[0, 0, r, c].item()) - x1_val) < 1e-6

    # --- Part A2: Closed-form jump at first visual milestone t = 50 (index 49) ---
    t50_idx = 49
    alpha_bar_50 = float(schedule.alphas_cumprod[t50_idx].item())
    sqrt_alpha_bar_50 = float(schedule.sqrt_alphas_cumprod[t50_idx].item())
    sqrt_one_minus_alpha_bar_50 = float(schedule.sqrt_one_minus_alphas_cumprod[t50_idx].item())
    x50_tensor, _ = q_sample_closed_form(
        x0_batch, torch.tensor([t50_idx], dtype=torch.long), schedule, noise=eps_tensor
    )
    x50_val = float(x50_tensor[0, 0, r, c].item())

    # --- Part B: Step-by-Step Loss Calculation L_simple at pixel (r, c) ---
    torch.manual_seed(config.SEED)
    unet = ConditionalUNet(in_channels=1, base_channels=32, time_dim=64)
    unet.eval()
    with torch.no_grad():
        eps_pred_tensor = unet(
            x1_tensor,
            torch.tensor([t1_idx], dtype=torch.long),
            torch.tensor([label], dtype=torch.long),
        )
    eps_pred_val = float(eps_pred_tensor[0, 0, r, c].item())
    diff_val = eps_val - eps_pred_val
    pointwise_loss_val = diff_val ** 2
    full_image_mse = float(torch.mean((eps_tensor - eps_pred_tensor) ** 2).item())

    results = {
        "seed": config.SEED,
        "digit_label": label,
        "pixel_coord": [r, c],
        "raw_pixel_0_255": raw_pixel_255,
        "x0": x0_val,
        "epsilon": eps_val,
        "t1": {
            "beta_1": beta_1,
            "alpha_1": alpha_1,
            "sqrt_one_minus_beta1": sqrt_one_minus_beta1,
            "sqrt_beta1": sqrt_beta1,
            "conditional_mean": cond_mean_t1,
            "noise_term": noise_term_t1,
            "x1": x1_val,
        },
        "t50_closed_form": {
            "alpha_bar_50": alpha_bar_50,
            "sqrt_alpha_bar_50": sqrt_alpha_bar_50,
            "sqrt_one_minus_alpha_bar_50": sqrt_one_minus_alpha_bar_50,
            "x50": x50_val,
        },
        "loss_l_simple": {
            "eps_true": eps_val,
            "eps_pred_untrained": eps_pred_val,
            "error_diff": diff_val,
            "pointwise_squared_error": pointwise_loss_val,
            "full_image_mse_t1": full_image_mse,
        },
    }

    json_path = config.TABLES_DIR / "step0_worked_example.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_report = f"""# Step 0 — Verified 1D Worked Numerical Example (Section 3.2)

- **Dataset & Sample**: MNIST Test Set, Digit `{label}`, Resolution $28 \\times 28$, Random Seed `{config.SEED}`
- **Selected Pixel Coordinate**: $(r, c) = ({r}, {c})$ on the bright stroke of digit `{label}`
- **Raw Grayscale Intensity**: $I_0 = {raw_pixel_255} \\in [0, 255]$
- **Normalized Clean Input ($x_0$)**:
  $$x_0 = \\frac{{{raw_pixel_255}}}{{127.5}} - 1 = {x0_val:.6f}$$
- **Sampled Standard Gaussian Noise ($\\epsilon \\sim \\mathcal{{N}}(0, 1)$ at pixel $({r}, {c})$)**:
  $$\\epsilon = {eps_val:.6f}$$

## Part A: Single-Step 1D Forward Noising ($t = 1$)
- **Variance Schedule at $t = 1$**: $\\beta_1 = {beta_1:.4f}$, $\\alpha_1 = 1 - \\beta_1 = {alpha_1:.4f}$
- **Signal Scaling Factor**: $\\sqrt{{1 - \\beta_1}} = \\sqrt{{{alpha_1:.4f}}} = {sqrt_one_minus_beta1:.6f}$
- **Noise Standard Deviation**: $\\sqrt{{\\beta_1}} = \\sqrt{{{beta_1:.4f}}} = {sqrt_beta1:.4f}$
- **Conditional Mean ($\\mu_1 = \\mathbb{{E}}[x_1 \\mid x_0]$)**:
  $$\\mu_1 = \\sqrt{{1 - \\beta_1}}\\, x_0 = {sqrt_one_minus_beta1:.6f} \\times {x0_val:.6f} = {cond_mean_t1:.6f}$$
- **Stochastic Noise Perturbation ($\\sqrt{{\\beta_1}}\\, \\epsilon$)**:
  $$\\sqrt{{\\beta_1}}\\, \\epsilon = {sqrt_beta1:.4f} \\times ({eps_val:.6f}) = {noise_term_t1:.6f}$$
- **Resulting Noisy Pixel State ($x_1$)**:
  $$x_1 = \\mu_1 + \\sqrt{{\\beta_1}}\\, \\epsilon = {cond_mean_t1:.6f} + ({noise_term_t1:.6f}) = {x1_val:.6f}$$

## Part A (Extension): Closed-Form Jump to $t = 50$ at the Same Pixel $({r}, {c})$
- **Cumulative Signal Product**: $\\bar{{\\alpha}}_{{50}} = {alpha_bar_50:.6f}$
- **Signal Coefficient**: $\\sqrt{{\\bar{{\\alpha}}_{{50}}}} = {sqrt_alpha_bar_50:.6f}$
- **Noise Coefficient**: $\\sqrt{{1 - \\bar{{\\alpha}}_{{50}}}} = {sqrt_one_minus_alpha_bar_50:.6f}$
- **Noisy State at $t = 50$**:
  $$x_{{50}} = {sqrt_alpha_bar_50:.6f} \\times ({x0_val:.6f}) + {sqrt_one_minus_alpha_bar_50:.6f} \\times ({eps_val:.6f}) = {x50_val:.6f}$$

## Part B: Step-by-Step Loss Calculation ($L_{{\\text{{simple}}}}$) at Pixel $({r}, {c})$
- **True Injected Noise ($\\epsilon$)**: ${eps_val:.6f}$
- **U-Net Predicted Noise ($\\epsilon_\\theta(x_1, 1)$)**: ${eps_pred_val:.6f}$
- **Prediction Residual ($\\epsilon - \\epsilon_\\theta(x_1, 1)$)**:
  $${eps_val:.6f} - ({eps_pred_val:.6f}) = {diff_val:.6f}$$
- **Pointwise Squared Error ($\\|\\epsilon - \\epsilon_\\theta(x_1, 1)\\|^2$)**:
  $$({diff_val:.6f})^2 = {pointwise_loss_val:.6f}$$
- **Full $28 \\times 28$ Image MSE Loss at $t = 1$**: ${full_image_mse:.6f}$
"""
    md_path = config.TABLES_DIR / "step0_worked_example.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(md_report)


if __name__ == "__main__":
    main()
