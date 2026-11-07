"""
Step 2: Pixel-Space DDPM Training & Sequential Reverse Denoising on 28x28 MNIST (Task 4b).
Supports:
  - Full 60,000 MNIST training set (via --full or config.TRAIN_MAX_SAMPLES = None)
  - Configurable sample sizes (e.g., --samples 2000 for high quality in ~3-4 mins on CPU)
  - Fast demo mode (via --fast for ultra-fast 25s CPU verification)

Generates the combined 2-row Figure (`fig_exp_pixel_forward_reverse.png`) for Section 4.1:
  - Row 1: Clean 28x28 image x_0 + 5 increasing forward noise levels
  - Row 2: Learned U-Net sequential reverse denoising trajectory (t = 1000 -> 0)
"""

import sys
import time
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset import get_mnist_datasets, get_canonical_digit_sample, unscale_to_0_1
from src.diffusion import (
    build_linear_schedule,
    q_sample_closed_form,
    compute_l_simple,
    p_sample_loop,
)
from src.models_unet import ConditionalUNet
from src.utils_metrics import (
    count_parameters,
    estimate_single_step_macs,
    compute_mse_psnr,
    save_figure_both,
    get_device,
    describe_device,
)


def build_balanced_mnist_subset(samples_per_class: int = 32) -> TensorDataset:
    """
    Create a clean, consistent balanced dataset of all 10 MNIST digits (0-9).
    Kept for backwards-compatibility and quick illustrative checks.
    """
    _, test_ds = get_mnist_datasets(download=True)
    prototypes = {}
    for idx in range(len(test_ds)):
        img, label = test_ds[idx]
        lbl = int(label)
        if lbl not in prototypes:
            prototypes[lbl] = img
        if len(prototypes) == 10:
            break

    xs, ys = [], []
    for lbl in range(10):
        for _ in range(samples_per_class):
            xs.append(prototypes[lbl])
            ys.append(lbl)
    return TensorDataset(torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long))


def get_training_dataloader(batch_size: int = 64, max_samples: int | None = None, seed: int = config.SEED):
    """
    Load training DataLoader.
    - If max_samples is None: loads FULL 60,000 MNIST training set.
    - If max_samples is an integer: extracts a representative subset of that size.
    """
    train_ds, _ = get_mnist_datasets(download=True)
    if max_samples is not None and 0 < max_samples < len(train_ds):
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_ds), generator=rng)[:max_samples].tolist()
        dataset = Subset(train_ds, indices)
    else:
        dataset = train_ds

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    return loader, dataset


def train_and_evaluate_pixel_ddpm(
    epochs: int = 20,
    batch_size: int = 64,
    max_samples: int | None = getattr(config, "TRAIN_MAX_SAMPLES", None),
    device: str | torch.device | None = None,
):
    torch.manual_seed(config.SEED)
    device = torch.device(device) if device is not None else get_device()
    print(f"[Pixel DDPM] Detected device: {describe_device(device)}")

    schedule = build_linear_schedule(
        T=config.T,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device,
    )

    loader, dataset = get_training_dataloader(batch_size=batch_size, max_samples=max_samples, seed=config.SEED)
    img_x0, label_3, (r, c) = get_canonical_digit_sample(digit=3)

    n_samples = len(dataset)
    is_full = (n_samples >= 60000)
    print("=" * 72)
    print(f"[Pixel DDPM] Training on {n_samples} images ({'FULL 60,000 MNIST' if is_full else f'Subset of {n_samples} images'})")
    print(f"             Device: {device} | Epochs: {epochs} | Batch size: {batch_size} | Batches/epoch: {len(loader)}")
    if is_full and device.type == "cpu":
        print("             NOTE: Running full 60,000 images on CPU will take significant time.")
        print("             (To run faster on CPU, you can use: python scripts/step2_train_ddpm.py --samples 2000)")
    print("=" * 72)

    torch.manual_seed(config.SEED)
    model = ConditionalUNet(
        in_channels=config.IMAGE_CHANNELS,
        base_channels=32,
        time_dim=64,
        num_classes=10,
        use_cross_attention=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    epoch_times = []
    final_loss = 0.0

    model.train()
    for ep in range(epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        running_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, config.T, (bx.shape[0],), device=device, dtype=torch.long)
            loss, _, _ = compute_l_simple(model, bx, t, schedule, y=by)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        lr_sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        epoch_times.append(ep_time)
        final_loss = running_loss / max(n_batches, 1)
        print(f"  Epoch {ep+1:02d}/{epochs:02d} | L_simple = {final_loss:.4f} | Time = {ep_time:.2f}s")

    ckpt_path = config.CHECKPOINTS_DIR / "ddpm_pixel_28x28.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Pixel DDPM] Saved checkpoint to: {ckpt_path}")

    # Evaluate L_simple at pixel (r, c) after training for comparison with Step 0
    model.eval()
    x0_batch = img_x0.unsqueeze(0).to(device)
    torch.manual_seed(config.SEED)
    # Generate noise on CPU then move to device, so the fixed epsilon matches Step 0/Step 1
    # bit-for-bit regardless of whether training runs on GPU or CPU.
    fixed_eps = torch.randn_like(x0_batch.cpu()).to(device)
    t1_tensor = torch.tensor([0], device=device, dtype=torch.long)
    x1_batch, _ = q_sample_closed_form(x0_batch, t1_tensor, schedule, noise=fixed_eps)
    with torch.no_grad():
        eps_pred_trained = model(x1_batch, t1_tensor, torch.tensor([label_3], device=device, dtype=torch.long))
    trained_eps_pred_rc = float(eps_pred_trained[0, 0, r, c].item())

    # Measure 1000-step sequential reverse denoising time for 1 sample
    print("[Pixel DDPM] Running sequential 1000-step reverse denoising...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_sample_start = time.perf_counter()
    x0_gen, rev_history = p_sample_loop(
        model=model,
        shape=(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE),
        schedule=schedule,
        y=torch.tensor([label_3], device=device, dtype=torch.long),
        steps_to_record=config.REVERSE_STEPS_SHOW,
        seed=config.SEED,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    sampling_time_1000 = time.perf_counter() - t_sample_start
    print(f"[Pixel DDPM] Completed 1000-step sampling in {sampling_time_1000:.2f}s ({config.T / sampling_time_1000:.1f} steps/s)")

    # Render combined 2-row Figure (`fig_exp_pixel_forward_reverse.png`)
    fwd_imgs = [unscale_to_0_1(x0_batch[0, 0].cpu())]
    fwd_titles = ["Clean $x_0$ ($t=0$)"]
    for step_idx, step_lbl in zip(config.FORWARD_STEPS_5, config.FORWARD_LABELS_5):
        xt_b, _ = q_sample_closed_form(
            x0_batch, torch.tensor([step_idx], device=device, dtype=torch.long), schedule, noise=fixed_eps
        )
        fwd_imgs.append(unscale_to_0_1(xt_b[0, 0].cpu()))
        fwd_titles.append(f"Forward ${step_lbl}$")

    rev_imgs = [unscale_to_0_1(rev_history[s][0, 0].cpu()) for s in config.REVERSE_STEPS_SHOW]
    rev_titles = [f"Reverse ${lbl}$" for lbl in config.REVERSE_LABELS_SHOW]

    fig, axes = plt.subplots(2, 6, figsize=(13.8, 4.9))
    for col in range(6):
        # Row 1: Forward Noising
        ax_f = axes[0, col]
        ax_f.imshow(fwd_imgs[col], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        # Highlight the worked-example pixel (r, c) from Section 3.2
        ax_f.add_patch(patches.Rectangle(
            (c - 0.5, r - 0.5), 1.0, 1.0, linewidth=1.1, edgecolor="#e11d48", facecolor="none"
        ))
        ax_f.set_title(fwd_titles[col], fontsize=9.5, pad=4)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        for sp in ax_f.spines.values():
            sp.set_edgecolor("#1e4e8c")
            sp.set_linewidth(0.9)

        # Row 2: Reverse Denoising
        ax_r = axes[1, col]
        ax_r.imshow(rev_imgs[col], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        # Highlight the worked-example pixel (r, c) from Section 3.2
        ax_r.add_patch(patches.Rectangle(
            (c - 0.5, r - 0.5), 1.0, 1.0, linewidth=1.1, edgecolor="#e11d48", facecolor="none"
        ))
        ax_r.set_title(rev_titles[col], fontsize=9.5, pad=4)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        for sp in ax_r.spines.values():
            sp.set_edgecolor("#b43c2f")
            sp.set_linewidth(0.9)

    axes[0, 0].set_ylabel("Row 1: Forward\n$q(x_t \\mid x_0)$", fontsize=9.5, fontweight="bold", color="#1e4e8c")
    axes[1, 0].set_ylabel("Row 2: Reverse\n$p_\\theta(x_{t-1} \\mid x_t)$", fontsize=9.5, fontweight="bold", color="#b43c2f")

    plt.tight_layout()
    fig_path = config.FIGURES_DIR / "fig_exp_pixel_forward_reverse.png"
    save_figure_both(fig, fig_path, also_pdf=True)
    print(f"[Pixel DDPM] Saved figure: {fig_path}")

    # Compute computational metrics
    macs_per_step = estimate_single_step_macs(model, (1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE), device=device)
    params_count = count_parameters(model)
    mse_val, psnr_val = compute_mse_psnr(unscale_to_0_1(x0_batch[0, 0].cpu()), unscale_to_0_1(x0_gen[0, 0].cpu()))

    stats = {
        "model": "Pixel-Space DDPM",
        "operating_space": "1 x 28 x 28 (784 dims)",
        "spatial_resolution": "28 x 28",
        "tensor_elements": 784,
        "unet_params": params_count,
        "macs_per_step": macs_per_step,
        "flops_per_step": macs_per_step * 2,
        "total_gflops_1000_steps": (macs_per_step * 2 * config.T) / 1e9,
        "avg_epoch_time_sec": float(sum(epoch_times) / len(epoch_times)),
        "sampling_time_1000_steps_sec": float(sampling_time_1000),
        "steps_per_sec": float(config.T / sampling_time_1000),
        "final_train_loss": final_loss,
        "trained_eps_pred_rc": trained_eps_pred_rc,
        "sample_mse_vs_canonical": mse_val,
        "sample_psnr_db": psnr_val,
        "training_samples": n_samples,
    }
    (config.TABLES_DIR / "ddpm_pixel_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train Pixel-Space DDPM on MNIST")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--samples", type=int, default=None, help="Max training samples (e.g. 2000, 5000, or None for full 60,000)")
    parser.add_argument("--full", action="store_true", help="Force train on FULL 60,000 MNIST dataset")
    parser.add_argument("--fast", action="store_true", help="Quick 320-sample demo mode (~25s on CPU)")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device: auto (GPU if available, else CPU), cuda, or cpu",
    )
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Pixel DDPM] CUDA requested but no GPU is available; falling back to CPU.")
        device = "cpu"

    if args.fast:
        max_samples = 320
    elif args.full:
        max_samples = None
    else:
        max_samples = args.samples  # if None, defaults to config.TRAIN_MAX_SAMPLES

    train_and_evaluate_pixel_ddpm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=max_samples,
        device=device,
    )


if __name__ == "__main__":
    main()
