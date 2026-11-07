"""
Step 1: Forward Noising at Five Progressive Noise Levels on a 28x28 Image (Task 4a).
Uses the exact same canonical 28x28 MNIST digit '3' and random seed (SEED=42)
as Step 0 (Section 3.2 worked example).
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset import get_canonical_digit_sample, unscale_to_0_1
from src.diffusion import build_linear_schedule, q_sample_closed_form
from src.utils_metrics import save_figure_both, get_device, describe_device


def generate_forward_5levels_figure(device: str | torch.device | None = None):
    torch.manual_seed(config.SEED)
    device = torch.device(device) if device is not None else get_device()
    print(f"[Forward Noising] Detected device: {describe_device(device)}")
    schedule = build_linear_schedule(
        T=config.T,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device,
    )

    img_x0, label, (r, c) = get_canonical_digit_sample(digit=3)
    x0_batch = img_x0.unsqueeze(0).to(device)  # [1, 1, 28, 28]

    torch.manual_seed(config.SEED)
    # Generate noise on CPU then move to device so it matches Step 0 bit-for-bit.
    fixed_eps = torch.randn_like(x0_batch.cpu()).to(device)

    images = [unscale_to_0_1(x0_batch[0, 0])]
    titles = [f"Clean $x_0$ ($t=0$)\n$\\sqrt{{\\bar{{\\alpha}}_0}}=1.00,\\; x_0({r},{c})={x0_batch[0,0,r,c]:.2f}$"]

    for step_idx, step_label in zip(config.FORWARD_STEPS_5, config.FORWARD_LABELS_5):
        t_tensor = torch.tensor([step_idx], dtype=torch.long, device=device)
        xt_batch, _ = q_sample_closed_form(x0_batch, t_tensor, schedule, noise=fixed_eps)
        sqrt_ab = float(schedule.sqrt_alphas_cumprod[step_idx].item())
        xt_pixel = float(xt_batch[0, 0, r, c].item())
        images.append(unscale_to_0_1(xt_batch[0, 0]))
        titles.append(f"${step_label}$\n$\\sqrt{{\\bar{{\\alpha}}_t}}={sqrt_ab:.2f},\\; x_t({r},{c})={xt_pixel:.2f}$")

    fig, axes = plt.subplots(1, len(images), figsize=(14.2, 2.7))
    for idx, (ax, img_arr, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img_arr, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        # Highlight the worked-example pixel (r, c) on Clean x_0 and t=50
        rect = patches.Rectangle(
            (c - 0.5, r - 0.5),
            1.0,
            1.0,
            linewidth=1.1,
            edgecolor="#e11d48" if idx == 0 else "#3b82f6",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title(title, fontsize=9.2, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#b0b8c4")
            spine.set_linewidth(0.8)

    fig.suptitle(
        f"Closed-Form Forward Noising $q(x_t \\mid x_0)$ at Five Noise Levels on $28 \\times 28$ MNIST (Seed={config.SEED}, Highlighted Pixel $({r},{c})$)",
        fontsize=10.5,
        y=1.03,
    )
    plt.tight_layout()
    out_png = config.FIGURES_DIR / "fig_forward_5levels.png"
    save_figure_both(fig, out_png, also_pdf=True)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forward-noising 5-level figure")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device: auto (GPU if available, else CPU), cuda, or cpu",
    )
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Forward Noising] CUDA requested but no GPU is available; falling back to CPU.")
        device = "cpu"
    generate_forward_5levels_figure(device=device)
