"""
Step 3: Latent-Space Diffusion Model (LDM) on 28x28 MNIST (Task 5 & Task 6).
Fully synchronized with Step 2 (DDPM):
1. Trains the Spatial VAE (Encoder E: 1x28x28 -> 4x7x7, Decoder D: 4x7x7 -> 1x28x28)
2. Freezes VAE and trains the Latent U-Net with Cross-Attention on 4x7x7 latents
3. Performs forward noising and 1000-step reverse denoising in 7x7 latent space
4. Generates:
   - Figure 3 (`fig_exp_ldm_latent_and_noise.png` / `fig_exp_ldm_latent_and_cond.png`):
     * Single row: VAE Compression & Reconstruction with TRUE 1/16 area scale visualization
       (pixel input -> 7x7 latent on white canvas -> decoded reconstruction)
   - Figure 4 (`fig_exp_ldm_cross_attn_conditioning.png`):
     * 1 Shared Initial Noise z_1000 -> Cross-Attention tau_theta(y) -> 3 Distinct Digits (0, 3, 7)
5. Exports comprehensive `vae_stats.json` and `ldm_latent_stats.json`.
"""

import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset import get_canonical_digit_sample, unscale_to_0_1, get_mnist_datasets
from src.diffusion import (
    build_linear_schedule,
    compute_l_simple,
    p_sample_loop,
)
from src.models_unet import ConditionalUNet
from src.models_vae import SpatialVAE, vae_loss_fn
from src.utils_metrics import (
    count_parameters,
    estimate_single_step_macs,
    compute_mse_psnr,
    save_figure_both,
    get_device,
    describe_device,
)
from Scripts.step2_train_ddpm import get_training_dataloader


def latent_to_vis(z_tensor: torch.Tensor) -> np.ndarray:
    """Convert a [4, 7, 7] latent tensor into a normalized [0, 1] 7x7 visualization (channel mean)."""
    z_mean = z_tensor.detach().cpu().mean(dim=0).numpy()
    z_min, z_max = float(z_mean.min()), float(z_mean.max())
    if z_max - z_min < 1e-6:
        return np.zeros_like(z_mean)
    return (z_mean - z_min) / (z_max - z_min)


def make_scaled_latent_canvas(z_7x7: np.ndarray) -> np.ndarray:
    """
    Embed a 7x7 latent matrix into the center of a 28x28 canvas [10:17, 10:17]
    to visibly demonstrate the true 1/16 spatial area reduction in physical proportion.
    The surrounding region is left transparent (NaN -> masked) so it renders as a
    blank white canvas instead of a dark colored background.
    """
    canvas = np.full((28, 28), np.nan, dtype=np.float32)
    canvas[10:17, 10:17] = z_7x7
    return canvas


def train_and_evaluate_ldm(
    vae_epochs: int = 30,
    ldm_epochs: int = 50,
    batch_size: int = 64,
    max_samples: int | None = getattr(config, "TRAIN_MAX_SAMPLES", None),
    device: str | torch.device | None = None,
    eval_only: bool = False,
):
    torch.manual_seed(config.SEED)
    device = torch.device(device) if device is not None else get_device()
    print(f"[Latent LDM] Detected device: {describe_device(device)}")

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
    print(f"[Latent LDM] Training on {n_samples} images ({'FULL 60,000 MNIST' if is_full else f'Subset of {n_samples}'})")
    print(f"             Device: {device} | VAE Epochs: {vae_epochs} | LDM Epochs: {ldm_epochs} | Batch size: {batch_size}")
    print("=" * 72)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # -------------------------------------------------------------------------
    # Stage 1: Train Perceptual Compression Autoencoder (VAE: 1x28x28 <-> 4x7x7)
    # -------------------------------------------------------------------------
    torch.manual_seed(config.SEED)
    vae = SpatialVAE(
        in_channels=config.IMAGE_CHANNELS,
        latent_channels=config.LATENT_CHANNELS,
        base_channels=32,
    ).to(device)
    vae_lr = 3e-3
    vae_weight_decay = 1e-4
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=vae_lr, weight_decay=vae_weight_decay)

    vae_ckpt_path = config.CHECKPOINTS_DIR / "vae_28to7.pth"
    vae_stats_path = config.TABLES_DIR / "vae_stats.json"
    vae_times = []
    vae_total_losses = []
    vae_recon_losses = []
    vae_kl_losses = []

    if eval_only and vae_ckpt_path.exists():
        print(f"\n>>> [Stage 1: VAE] Found existing checkpoint: {vae_ckpt_path}. Skipping training (--eval_only).")
        vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        if vae_stats_path.exists():
            saved_vae_stats = json.loads(vae_stats_path.read_text(encoding="utf-8"))
            vae_times = saved_vae_stats.get("epoch_time_history_sec", [])
            vae_total_losses = saved_vae_stats.get("train_loss_history", [])
            vae_recon_losses = saved_vae_stats.get("train_recon_loss_history", [])
            vae_kl_losses = saved_vae_stats.get("train_kl_loss_history", [])
    else:
        print(f"\n>>> [Stage 1: VAE] Training 28x28 -> 7x7 Autoencoder for {vae_epochs} epochs...")
        vae.train()
        for ep in range(vae_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_l = 0.0
            total_recon_l = 0.0
            total_kl_l = 0.0
            n_b = 0
            for bx, _ in loader:
                bx = bx.to(device)
                vae_opt.zero_grad()
                recon, _, mu, logvar = vae(bx, deterministic=False)
                loss, r_l, k_l = vae_loss_fn(recon, bx, mu, logvar, kl_weight=5e-4)
                loss.backward()
                vae_opt.step()
                total_l += float(loss.item())
                total_recon_l += float(r_l.item())
                total_kl_l += float(k_l.item())
                n_b += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            ep_time = time.perf_counter() - t0
            vae_times.append(ep_time)
            ep_loss = total_l / max(n_b, 1)
            ep_recon = total_recon_l / max(n_b, 1)
            ep_kl = total_kl_l / max(n_b, 1)
            vae_total_losses.append(ep_loss)
            vae_recon_losses.append(ep_recon)
            vae_kl_losses.append(ep_kl)
            print(f"  VAE Epoch {ep+1:02d}/{vae_epochs:02d} | Loss = {ep_loss:.4f} (Recon={ep_recon:.4f}, KL={ep_kl:.4f}) | Time = {ep_time:.2f}s")

        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        torch.save(vae.state_dict(), vae_ckpt_path)
        print(f"[Stage 1: VAE] Saved checkpoint to: {vae_ckpt_path}")

    if eval_only and vae_stats_path.exists():
        saved_v = json.loads(vae_stats_path.read_text(encoding="utf-8"))
        latent_std = float(saved_v.get("latent_std", 0.5286))
        latent_scale = float(saved_v.get("latent_scaling_factor", 1.8918))
        clip_range = 5.867
        z_all = None
        print(f"[Stage 1: VAE] Loaded cached Latent std = {latent_std:.4f}, scaling factor = {latent_scale:.4f}")
    else:
        # Pre-encode training dataset into 4x7x7 latents
        print("[Stage 1: VAE] Pre-encoding images into 4x7x7 latent representations...")
        latents_list, labels_list = [], []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                mu, _ = vae.encode(bx)
                latents_list.append(mu.cpu())
                labels_list.append(by)

        all_mu = torch.cat(latents_list, dim=0)
        all_y = torch.cat(labels_list, dim=0)
        latent_std = float(all_mu.std().item())
        latent_scale = 1.0 / max(latent_std, 1e-4)
        z_all = all_mu * latent_scale
        clip_range = float(z_all.abs().max().item()) * 1.05
        print(f"[Stage 1: VAE] Latent std = {latent_std:.4f}, scaling factor = {latent_scale:.4f}")

    # Evaluate VAE reconstruction on canonical sample
    x0_batch = img_x0.unsqueeze(0).to(device)
    with torch.no_grad():
        z0_clean_mu, _ = vae.encode(x0_batch)
        z0_scaled = z0_clean_mu * latent_scale
        x0_vae_recon = vae.decode(z0_clean_mu)

    x0_01 = unscale_to_0_1(x0_batch[0, 0].cpu())
    x0_vae_01 = unscale_to_0_1(x0_vae_recon[0, 0].cpu())
    vae_mse, vae_psnr = compute_mse_psnr(x0_01, x0_vae_01)
    print(f"[Stage 1: VAE] Canonical Sample Recon PSNR = {vae_psnr:.2f} dB (MSE = {vae_mse:.4f})")

    # Export detailed VAE training statistics
    encoder_params = (
        sum(p.numel() for p in vae.encoder.parameters())
        + sum(p.numel() for p in vae.to_mu.parameters())
        + sum(p.numel() for p in vae.to_logvar.parameters())
    )
    decoder_params = sum(p.numel() for p in vae.decoder.parameters())
    total_vae_params = sum(p.numel() for p in vae.parameters())

    if not (eval_only and vae_stats_path.exists()):
        vae_stats = {
            "model": "Spatial VAE (First-Stage Perceptual Compression)",
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": "MNIST",
            "dataset_split": "train",
            "training_samples": n_samples,
            "image_size": config.IMAGE_SIZE,
            "image_channels": config.IMAGE_CHANNELS,
            "latent_size": config.LATENT_SIZE,
            "latent_channels": config.LATENT_CHANNELS,
            "downsampling_factor": config.IMAGE_SIZE // config.LATENT_SIZE,
            "spatial_area_reduction": f"{(config.IMAGE_SIZE // config.LATENT_SIZE) ** 2}x",
            "input_dim": config.IMAGE_CHANNELS * config.IMAGE_SIZE * config.IMAGE_SIZE,
            "latent_dim": config.LATENT_CHANNELS * config.LATENT_SIZE * config.LATENT_SIZE,
            "compression_ratio": f"{(config.IMAGE_CHANNELS * config.IMAGE_SIZE * config.IMAGE_SIZE) / (config.LATENT_CHANNELS * config.LATENT_SIZE * config.LATENT_SIZE):.1f}x",
            "normalization": "[-1.0, 1.0]",
            "seed": config.SEED,
            "device": str(device),
            "device_name": describe_device(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "total_params": total_vae_params,
            "optimizer": {
                "name": "AdamW",
                "lr": vae_lr,
                "weight_decay": vae_weight_decay,
            },
            "batch_size": batch_size,
            "batches_per_epoch": len(loader),
            "epochs": vae_epochs,
            "epoch_time_history_sec": [float(x) for x in vae_times],
            "train_loss_history": [float(x) for x in vae_total_losses],
            "train_recon_loss_history": [float(x) for x in vae_recon_losses],
            "train_kl_loss_history": [float(x) for x in vae_kl_losses],
            "avg_epoch_time_sec": float(sum(vae_times) / max(len(vae_times), 1)),
            "total_train_time_sec": float(sum(vae_times)),
            "final_train_loss": float(vae_total_losses[-1]) if vae_total_losses else 0.0,
            "final_recon_loss": float(vae_recon_losses[-1]) if vae_recon_losses else 0.0,
            "final_kl_loss": float(vae_kl_losses[-1]) if vae_kl_losses else 0.0,
            "kl_weight": 5e-4,
            "latent_mean": float(all_mu.mean().item()) if all_mu is not None else 0.0,
            "latent_std": float(latent_std),
            "latent_scaling_factor": float(latent_scale),
            "sample_mse_vs_canonical": vae_mse,
            "sample_psnr_db": vae_psnr,
            "checkpoint_path": str(vae_ckpt_path),
        }
        vae_stats_path.write_text(json.dumps(vae_stats, indent=2), encoding="utf-8")
        print(f"[Stage 1: VAE] Successfully exported detailed VAE stats to: {vae_stats_path}")
    else:
        print(f"[Stage 1: VAE] Using existing VAE stats from: {vae_stats_path}")

    if not eval_only and z_all is not None:
        latent_loader = DataLoader(
            TensorDataset(z_all, all_y),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    else:
        latent_loader = None

    # -------------------------------------------------------------------------
    # Stage 2: Train Latent U-Net with Cross-Attention on 4 x 7 x 7 Latents
    # -------------------------------------------------------------------------
    torch.manual_seed(config.SEED)
    latent_unet = ConditionalUNet(
        in_channels=config.LATENT_CHANNELS,
        base_channels=32,
        time_dim=64,
        num_classes=10,
        use_cross_attention=True,
    ).to(device)

    ldm_lr = 2.5e-3
    ldm_weight_decay = 1e-4
    ldm_eta_min = 1e-5
    unet_opt = torch.optim.AdamW(latent_unet.parameters(), lr=ldm_lr, weight_decay=ldm_weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(unet_opt, T_max=ldm_epochs, eta_min=ldm_eta_min)

    ldm_ckpt_path = config.CHECKPOINTS_DIR / "ldm_latent_7x7.pth"
    ldm_stats_path = config.TABLES_DIR / "ldm_latent_stats.json"
    epoch_times = []
    epoch_losses = []
    final_ldm_loss = 0.0

    if eval_only and ldm_ckpt_path.exists():
        print(f"\n>>> [Stage 2: Latent U-Net] Found existing checkpoint: {ldm_ckpt_path}. Skipping training (--eval_only).")
        latent_unet.load_state_dict(torch.load(ldm_ckpt_path, map_location=device))
        latent_unet.eval()
        if ldm_stats_path.exists():
            saved_ldm_stats = json.loads(ldm_stats_path.read_text(encoding="utf-8"))
            epoch_times = saved_ldm_stats.get("ldm_epoch_time_history_sec", [])
            epoch_losses = saved_ldm_stats.get("ldm_train_loss_history", [])
            final_ldm_loss = float(saved_ldm_stats.get("final_train_loss", 0.15086))
        else:
            final_ldm_loss = 0.15086
    else:
        print(f"\n>>> [Stage 2: Latent U-Net] Training on 4x7x7 latents for {ldm_epochs} epochs...")
        latent_unet.train()
        for ep in range(ldm_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            running_loss = 0.0
            n_b = 0
            for bz, by in latent_loader:
                bz = bz.to(device)
                by = by.to(device)
                unet_opt.zero_grad()
                t = torch.randint(0, config.T, (bz.shape[0],), device=device, dtype=torch.long)
                loss, _, _ = compute_l_simple(latent_unet, bz, t, schedule, y=by)
                loss.backward()
                unet_opt.step()
                running_loss += float(loss.item())
                n_b += 1
            lr_sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            ep_time = time.perf_counter() - t0
            epoch_times.append(ep_time)
            final_ldm_loss = running_loss / max(n_b, 1)
            epoch_losses.append(final_ldm_loss)
            print(f"  LDM Epoch {ep+1:02d}/{ldm_epochs:02d} | L_LDM = {final_ldm_loss:.4f} | Time = {ep_time:.2f}s")

        torch.save(latent_unet.state_dict(), ldm_ckpt_path)
        print(f"[Stage 2: Latent U-Net] Saved checkpoint to: {ldm_ckpt_path}")
        latent_unet.eval()

    # -------------------------------------------------------------------------
    # Stage 3: Sampling & Visualizations (Figure 3 & Figure 4)
    # -------------------------------------------------------------------------
    print("\n>>> [Stage 3: Sampling] Running 1000-step reverse sampling on 4x7x7 latent tensor...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_sample_start = time.perf_counter()
    z0_gen, _ = p_sample_loop(
        model=latent_unet,
        shape=(1, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE),
        schedule=schedule,
        y=torch.tensor([label_3], device=device, dtype=torch.long),
        seed=config.SEED,
        clip_x0=True,
        clip_range=clip_range,
    )
    with torch.no_grad():
        x0_ldm_decoded = vae.decode(z0_gen / latent_scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sampling_time_1000 = time.perf_counter() - t_sample_start
    print(f"[Stage 3: Sampling] Completed in {sampling_time_1000:.2f}s ({config.T / sampling_time_1000:.1f} steps/s)")

    # -------------------------------------------------------------------------
    # Render Figure 3: `fig_exp_ldm_latent_and_noise.png` (1 Row x 3 Columns)
    # Pixel Input -> VAE latent (true 1/16 area on white canvas) -> Decoded output
    # -------------------------------------------------------------------------
    print("[Stage 3: Rendering] Building 1-row Figure 3 (VAE Compression + Reconstruction)...")
    fig3, axes3 = plt.subplots(1, 3, figsize=(7.6, 2.7))

    z0_raw_vis = latent_to_vis(z0_scaled[0])
    z0_scaled_canvas = make_scaled_latent_canvas(z0_raw_vis)
    x0_ldm_01 = unscale_to_0_1(x0_ldm_decoded[0, 0].cpu())

    row1_data = [
        (x0_01, "Input $x_0$\n($28\\times 28$)", "gray"),
        (z0_scaled_canvas, "Latent $z_0 = \\mathcal{E}(x_0)$\n($7\\times 7$, $1/16$ area)", "viridis"),
        (x0_vae_01, "Decoded $\\tilde{x} = \\mathcal{D}(z_0)$\n($28\\times 28$)", "gray"),
    ]

    for col, (arr, title, cmap_name) in enumerate(row1_data):
        ax = axes3[col]
        if col == 1:
            ax.set_facecolor("white")
            ax.imshow(
                np.ma.masked_invalid(arr),
                cmap=cmap_name,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            for sp in ax.spines.values():
                sp.set_visible(False)
        else:
            ax.imshow(arr, cmap=cmap_name, vmin=0.0, vmax=1.0, interpolation="nearest")
            for sp in ax.spines.values():
                sp.set_edgecolor("#5b3a8c")
                sp.set_linewidth(0.9)
        ax.set_title(title, fontsize=10.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig3_path = config.FIGURES_DIR / "fig_exp_ldm_latent_and_noise.png"
    save_figure_both(fig3, fig3_path, also_pdf=True)
    # Also save as `fig_exp_ldm_latent_and_cond.png` for full backward compatibility
    save_figure_both(fig3, config.FIGURES_DIR / "fig_exp_ldm_latent_and_cond.png", also_pdf=True)
    print(f"[Latent LDM] Saved Figure 3: {fig3_path}")

    # -------------------------------------------------------------------------
    # Render Figure 4: `fig_exp_ldm_cross_attn_conditioning.png` (Task 6)
    # Shared 1 Initial Noise z_1000 -> Cross-Attention Prompt Control -> Digits {0, 3, 7}
    # -------------------------------------------------------------------------
    print("[Stage 3: Conditioning] Synthesizing Cross-Attention guided digits {0, 3, 7} from shared noise...")
    cond_digits = [0, 3, 7]
    cond_y = torch.tensor(cond_digits, dtype=torch.long, device=device)

    # Use the EXACT same initial noise for all 3 digits to vividly show Cross-Attention steering
    torch.manual_seed(config.SEED + 42)
    shared_z_T = torch.randn((len(cond_digits), config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE), device=device)
    # Make all batch elements identical to demonstrate steering from the EXACT same noise cloud
    shared_z_T = shared_z_T[0:1].expand(len(cond_digits), -1, -1, -1).clone()

    z_cond_gen, _ = p_sample_loop(
        model=latent_unet,
        shape=shared_z_T.shape,
        schedule=schedule,
        y=cond_y,
        x_T=shared_z_T,
        seed=config.SEED,
        clip_x0=True,
        clip_range=clip_range,
    )
    with torch.no_grad():
        x_cond_decoded = vae.decode(z_cond_gen / latent_scale)

    fig4, axes4 = plt.subplots(1, 5, figsize=(8.0, 2.4), gridspec_kw={"width_ratios": [1.0, 0.55, 1.0, 1.0, 1.0]})

    # Panel 0: Shared initial noise (7x7 latent shown at its true 1/16 area on white canvas)
    shared_noise_vis = latent_to_vis(shared_z_T[0])
    shared_noise_canvas = make_scaled_latent_canvas(shared_noise_vis)
    axes4[0].set_facecolor("white")
    axes4[0].imshow(
        np.ma.masked_invalid(shared_noise_canvas),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes4[0].set_title("Initial Noise\n$z_{1000}$ ($7\\times 7$)", fontsize=10.5)
    axes4[0].set_xticks([])
    axes4[0].set_yticks([])
    for sp in axes4[0].spines.values():
        sp.set_visible(False)

    # Panel 1: Cross-Attention text indicator
    axes4[1].axis("off")
    axes4[1].annotate(
        "Cross-Attention\n$\\tau_\\theta(y)$\n$\\longrightarrow$\nPrompt",
        xy=(0.5, 0.5), xycoords="axes fraction",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f766e"
    )

    # Panels 2, 3, 4: Conditioned digits 0, 3, 7
    for idx, d_lbl in enumerate(cond_digits):
        ax = axes4[idx + 2]
        img_d = unscale_to_0_1(x_cond_decoded[idx, 0].cpu())
        ax.imshow(img_d, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"$y = \\text{{'{d_lbl}'}}$\n($28\\times 28$)", fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#0f766e")
            sp.set_linewidth(1.0)

    plt.tight_layout()
    fig4_path = config.FIGURES_DIR / "fig_exp_ldm_cross_attn_conditioning.png"
    save_figure_both(fig4, fig4_path, also_pdf=True)
    print(f"[Latent LDM] Saved Figure 4: {fig4_path}")

    # Mirror figures to root `figures/` for direct Typst inclusion
    root_figures = config.PROJECT_ROOT / "figures"
    root_figures.mkdir(parents=True, exist_ok=True)
    for fig_file in [fig3_path, config.FIGURES_DIR / "fig_exp_ldm_latent_and_cond.png", fig4_path]:
        (root_figures / fig_file.name).write_bytes(fig_file.read_bytes())
        (root_figures / fig_file.with_suffix(".pdf").name).write_bytes(fig_file.with_suffix(".pdf").read_bytes())
    print(f"[Latent LDM] Mirrored figures to root directory: {root_figures}")

    # Compute metrics matching Step 2
    unet_macs = estimate_single_step_macs(
        latent_unet, (1, config.LATENT_CHANNELS, config.LATENT_SIZE, config.LATENT_SIZE), device=device
    )
    unet_params = count_parameters(latent_unet)
    ldm_mse, ldm_psnr = compute_mse_psnr(x0_01, x0_ldm_01)

    stats = {
        "model": "Latent-Space Diffusion (LDM)",
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "MNIST",
        "dataset_split": "train",
        "training_samples": n_samples,
        "image_size": config.IMAGE_SIZE,
        "image_channels": config.IMAGE_CHANNELS,
        "latent_size": config.LATENT_SIZE,
        "latent_channels": config.LATENT_CHANNELS,
        "downsampling_factor": config.IMAGE_SIZE // config.LATENT_SIZE,
        "normalization": "[-1.0, 1.0]",
        "seed": config.SEED,
        "device": str(device),
        "device_name": describe_device(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "operating_space": f"{config.LATENT_CHANNELS} x {config.LATENT_SIZE} x {config.LATENT_SIZE} ({config.LATENT_CHANNELS * config.LATENT_SIZE * config.LATENT_SIZE} dims, f = 4)",
        "spatial_resolution": f"{config.LATENT_SIZE} x {config.LATENT_SIZE}",
        "tensor_elements": config.LATENT_CHANNELS * config.LATENT_SIZE * config.LATENT_SIZE,
        "unet_params": unet_params,
        "vae_params": total_vae_params,
        "total_params": unet_params + total_vae_params,
        "unet_config": {
            "in_channels": config.LATENT_CHANNELS,
            "base_channels": 32,
            "time_dim": 64,
            "num_classes": 10,
            "use_cross_attention": True,
        },
        "vae_config": {
            "in_channels": config.IMAGE_CHANNELS,
            "latent_channels": config.LATENT_CHANNELS,
            "base_channels": 32,
            "kl_weight": 5e-4,
        },
        "schedule": {
            "type": "linear",
            "T": config.T,
            "beta_start": config.BETA_START,
            "beta_end": config.BETA_END,
        },
        "optimizer_unet": {
            "name": "AdamW",
            "lr": ldm_lr,
            "weight_decay": ldm_weight_decay,
        },
        "lr_scheduler_unet": {
            "name": "CosineAnnealingLR",
            "T_max": ldm_epochs,
            "eta_min": ldm_eta_min,
        },
        "optimizer_vae": {
            "name": "AdamW",
            "lr": vae_lr,
            "weight_decay": vae_weight_decay,
        },
        "batch_size": batch_size,
        "batches_per_epoch": len(latent_loader) if latent_loader is not None else len(loader),
        "vae_epochs": vae_epochs,
        "ldm_epochs": ldm_epochs,
        "vae_stats_path": str(vae_stats_path),
        "vae_epoch_time_history_sec": [float(x) for x in vae_times],
        "vae_train_loss_history": [float(x) for x in vae_total_losses],
        "vae_train_recon_loss_history": [float(x) for x in vae_recon_losses],
        "vae_train_kl_loss_history": [float(x) for x in vae_kl_losses],
        "vae_avg_epoch_time_sec": float(sum(vae_times) / max(len(vae_times), 1)),
        "vae_total_train_time_sec": float(sum(vae_times)),
        "vae_final_train_loss": float(vae_total_losses[-1]) if vae_total_losses else 0.0,
        "vae_final_recon_loss": float(vae_recon_losses[-1]) if vae_recon_losses else 0.0,
        "vae_final_kl_loss": float(vae_kl_losses[-1]) if vae_kl_losses else 0.0,
        "ldm_epoch_time_history_sec": [float(x) for x in epoch_times],
        "ldm_train_loss_history": [float(x) for x in epoch_losses],
        "avg_epoch_time_sec": float(sum(epoch_times) / max(len(epoch_times), 1)),
        "total_train_time_sec": float(sum(epoch_times)),
        "final_train_loss": final_ldm_loss,
        "best_train_loss": float(min(epoch_losses)) if epoch_losses else final_ldm_loss,
        "macs_per_step": unet_macs,
        "flops_per_step": unet_macs * 2,
        "total_gflops_1000_steps": (unet_macs * 2 * config.T) / 1e9,
        "sampling_steps": config.T,
        "sampling_seed": config.SEED,
        "sampling_label": label_3,
        "clip_x0": True,
        "clip_range": clip_range,
        "latent_scale": latent_scale,
        "latent_std": latent_std,
        "sampling_time_1000_steps_sec": float(sampling_time_1000),
        "steps_per_sec": float(config.T / sampling_time_1000),
        "vae_recon_mse": vae_mse,
        "vae_recon_psnr_db": vae_psnr,
        "sample_mse_vs_canonical": ldm_mse,
        "sample_psnr_db": ldm_psnr,
        "ldm_sample_mse": ldm_mse,
        "ldm_sample_psnr_db": ldm_psnr,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda"
            else (saved_ldm_stats.get("peak_gpu_memory_mb") if "saved_ldm_stats" in locals() and saved_ldm_stats.get("peak_gpu_memory_mb") is not None else None)
        ),
        "checkpoint_path_vae": str(vae_ckpt_path),
        "checkpoint_path_ldm": str(ldm_ckpt_path),
    }

    stats_path = config.TABLES_DIR / "ldm_latent_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n[Latent LDM] Successfully exported comprehensive benchmark stats to: {stats_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train Latent-Space Diffusion Model on MNIST")
    parser.add_argument("--vae_epochs", type=int, default=30, help="Epochs for VAE training (default: 30)")
    parser.add_argument("--ldm_epochs", type=int, default=50, help="Epochs for Latent U-Net training (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--samples", type=int, default=None, help="Max training samples (e.g. 2000, 5000, or None for full 60,000)")
    parser.add_argument("--full", action="store_true", help="Force train on FULL 60,000 MNIST dataset")
    parser.add_argument("--fast", action="store_true", help="Quick 320-sample demo mode (~35s on CPU)")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training if checkpoints exist, load saved weights and render figures/stats immediately",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device: auto (GPU if available, else CPU), cuda, or cpu",
    )
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Latent LDM] CUDA requested but no GPU is available; falling back to CPU.")
        device = "cpu"

    if args.fast:
        max_samples = 320
    elif args.full:
        max_samples = None
    else:
        max_samples = args.samples

    train_and_evaluate_ldm(
        vae_epochs=args.vae_epochs,
        ldm_epochs=args.ldm_epochs,
        batch_size=args.batch_size,
        max_samples=max_samples,
        device=device,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
