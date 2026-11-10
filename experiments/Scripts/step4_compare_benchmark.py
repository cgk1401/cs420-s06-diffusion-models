"""
CS420 Artificial Intelligence Seminar - Topic [S06]
Diffusion Models: From Adding Noise to Generating Images
Step 4: Benchmark Aggregator and Comparative Visualization (DDPM vs. LDM)

Loads existing empirical run stats from:
  - experiments/outputs/tables/ddpm_pixel_stats.json
  - experiments/outputs/tables/ldm_latent_stats.json
  - experiments/outputs/tables/vae_stats.json

Generates:
  1. experiments/outputs/tables/pixel_vs_latent_benchmark.md
  2. figures/fig_speed_flops_comparison.png & .pdf
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = PROJECT_ROOT / "experiments" / "outputs" / "tables"
EXP_FIGURES_DIR = PROJECT_ROOT / "experiments" / "outputs" / "figures"
ROOT_FIGURES_DIR = PROJECT_ROOT / "figures"


def main():
    print("=" * 72)
    print("[Step 4: Benchmark] Aggregating Pixel vs. Latent Diffusion Benchmarks...")
    print("=" * 72)

    ddpm_file = TABLES_DIR / "ddpm_pixel_stats.json"
    ldm_file = TABLES_DIR / "ldm_latent_stats.json"
    vae_file = TABLES_DIR / "vae_stats.json"

    if not ddpm_file.exists():
        raise FileNotFoundError(f"Missing DDPM benchmark file: {ddpm_file}")
    if not ldm_file.exists():
        raise FileNotFoundError(f"Missing LDM benchmark file: {ldm_file}")
    if not vae_file.exists():
        raise FileNotFoundError(f"Missing VAE benchmark file: {vae_file}")

    with open(ddpm_file, "r", encoding="utf-8") as f:
        ddpm_stats = json.load(f)
    with open(ldm_file, "r", encoding="utf-8") as f:
        ldm_stats = json.load(f)
    with open(vae_file, "r", encoding="utf-8") as f:
        vae_stats = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Extract and normalize benchmark metrics
    #    Every value is read directly from the recorded run stats. A missing
    #    field raises KeyError instead of silently falling back to a stale
    #    default, so the report can never contain an out-of-date number.
    # -------------------------------------------------------------------------
    ddpm_flops = ddpm_stats["total_gflops_1000_steps"]
    ldm_flops = ldm_stats["total_gflops_1000_steps"]
    flops_reduction = ddpm_flops / ldm_flops

    ddpm_vram = ddpm_stats["peak_gpu_memory_mb"]
    ldm_vram = ldm_stats["peak_gpu_memory_mb"]
    vram_reduction = ddpm_vram / ldm_vram

    ddpm_epoch_sec = ddpm_stats["avg_epoch_time_sec"]
    ldm_epoch_sec = ldm_stats["avg_epoch_time_sec"]
    speed_speedup = (ddpm_epoch_sec - ldm_epoch_sec) / ddpm_epoch_sec * 100.0
    throughput_ratio = ddpm_epoch_sec / ldm_epoch_sec

    ddpm_train_total_sec = ddpm_stats["total_train_time_sec"]
    ldm_train_total_sec = ldm_stats["total_train_time_sec"]
    vae_train_total_sec = vae_stats["total_train_time_sec"]
    ldm_pipeline_total_sec = ldm_train_total_sec + vae_train_total_sec
    pipeline_speedup = ddpm_train_total_sec / ldm_pipeline_total_sec

    ddpm_sample_sec = ddpm_stats["sampling_time_1000_steps_sec"]
    ldm_sample_sec = ldm_stats["sampling_time_1000_steps_sec"]
    ddpm_steps_per_sec = ddpm_stats["steps_per_sec"]
    ldm_steps_per_sec = ldm_stats["steps_per_sec"]

    ddpm_loss = ddpm_stats["final_train_loss"]
    ldm_loss = ldm_stats["final_train_loss"]

    vae_psnr = vae_stats["sample_psnr_db"]
    vae_mse = vae_stats["sample_mse_vs_canonical"]

    ddpm_params = ddpm_stats["unet_params"]
    ldm_unet_params = ldm_stats["unet_params"]
    vae_params = ldm_stats["vae_params"]
    ldm_total_params = ldm_stats["total_params"]

    ddpm_sample_psnr = ddpm_stats["sample_psnr_db"]
    ldm_sample_psnr = ldm_stats["sample_psnr_db"]

    print(f"  [Compute] DDPM GFLOPs: {ddpm_flops:.2f} | LDM GFLOPs: {ldm_flops:.2f} ({flops_reduction:.1f}x reduction)")
    print(f"  [Memory]  DDPM VRAM:   {ddpm_vram:.1f} MB | LDM VRAM:   {ldm_vram:.1f} MB ({vram_reduction:.1f}x reduction)")
    print(f"  [Speed]   DDPM Epoch:  {ddpm_epoch_sec:.2f}s | LDM Epoch: {ldm_epoch_sec:.2f}s ({speed_speedup:.1f}% faster)")
    print(f"  [Cost]    LDM pipeline: {ldm_pipeline_total_sec:.1f}s (U-Net {ldm_train_total_sec:.1f}s + VAE {vae_train_total_sec:.1f}s) vs DDPM {ddpm_train_total_sec:.1f}s")

    # -------------------------------------------------------------------------
    # 2. Render Figure 5: Speed and FLOPs Comparison Chart
    # -------------------------------------------------------------------------
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    categories = ["Pixel DDPM\n(28×28)", "Latent LDM\n(4×7×7)"]
    c_ddpm = "#1e4e8c"
    c_ldm = "#0f766e"

    # Subplot 1: GFLOPs (1000 steps)
    ax = axes[0]
    bars1 = ax.bar(categories, [ddpm_flops, ldm_flops], color=[c_ddpm, c_ldm], width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("GFLOPs (1000 Steps)", fontsize=9.5, fontweight="bold")
    ax.set_title("Computational Cost\n(FLOPs per Generation)", fontsize=10.2, pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.annotate(f"▼ {flops_reduction:.1f}× Reduction", xy=(1, ddpm_flops * 0.45), ha="center",
                fontsize=9.2, fontweight="bold", color="#b43c2f",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.95))
    ax.set_ylim(0, max(ddpm_flops, ldm_flops) * 1.25)

    # Subplot 2: Peak Memory (VRAM MB)
    ax = axes[1]
    bars2 = ax.bar(categories, [ddpm_vram, ldm_vram], color=[c_ddpm, c_ldm], width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Peak VRAM (MB)", fontsize=9.5, fontweight="bold")
    ax.set_title("Peak GPU Memory\n(Allocation Footprint)", fontsize=10.2, pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.annotate(f"▼ {vram_reduction:.1f}× Lower", xy=(1, ddpm_vram * 0.45), ha="center",
                fontsize=9.2, fontweight="bold", color="#b43c2f",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.95))
    ax.set_ylim(0, max(ddpm_vram, ldm_vram) * 1.25)

    # Subplot 3: Training Time (Seconds / Epoch)
    ax = axes[2]
    bars3 = ax.bar(categories, [ddpm_epoch_sec, ldm_epoch_sec], color=[c_ddpm, c_ldm], width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Seconds / Epoch", fontsize=9.5, fontweight="bold")
    ax.set_title("Training Throughput\n(Per-Epoch Latency)", fontsize=10.2, pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars3:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.annotate(f"▲ {speed_speedup:.0f}% Faster", xy=(1, ddpm_epoch_sec * 0.84), ha="center",
                fontsize=9.2, fontweight="bold", color="#0f766e",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ccfbf1", edgecolor="#0f766e", alpha=0.95))
    ax.set_ylim(0, max(ddpm_epoch_sec, ldm_epoch_sec) * 1.30)

    for ax in axes:
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # Save to both experiments/outputs/figures/ and figures/
    EXP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_png_exp = EXP_FIGURES_DIR / "fig_speed_flops_comparison.png"
    fig_pdf_exp = EXP_FIGURES_DIR / "fig_speed_flops_comparison.pdf"
    fig.savefig(fig_png_exp, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf_exp, bbox_inches="tight")

    fig_png_root = ROOT_FIGURES_DIR / "fig_speed_flops_comparison.png"
    fig_pdf_root = ROOT_FIGURES_DIR / "fig_speed_flops_comparison.pdf"
    fig_png_root.write_bytes(fig_png_exp.read_bytes())
    fig_pdf_root.write_bytes(fig_pdf_exp.read_bytes())

    print(f"[Step 4: Benchmark] Saved comparison chart to: {fig_png_exp} & {fig_png_root}")

    # -------------------------------------------------------------------------
    # 3. Generate Markdown Benchmark Table
    # -------------------------------------------------------------------------
    md_content = f"""# Diffusion Models Benchmark: Pixel-Space (DDPM) vs. Latent-Space (LDM)

This benchmark provides an empirical, reproducible comparison between **Pixel-Space DDPM** (Ho et al., 2020) and **Latent Diffusion Models - LDM** (Rombach et al., 2022) conducted on the full MNIST dataset ($N = 60,000$, 50 epochs, seed 42) along with cited literature results.

## 1. Summary Comparison Table (Section 4.2.3 / Table 3)

| Explicit Criterion | Pixel-Space Diffusion (DDPM) | Latent Diffusion (LDM) | Literature Benchmark (Ho 2020 / Rombach 2022) |
| :--- | :--- | :--- | :--- |
| **Operating Space & Dimensions** | Direct pixel space: $1 \\times 28 \\times 28 = 784$ scalar pixels | Latent space: $4 \\times 7 \\times 7 = 196$ elements ($f=4$, $16\\times$ area reduction via Spatial VAE) | ImageNet $256 \\times 256 \\times 3 = 196,608$ (DDPM) vs. $3 \\times 64 \\times 64 = 16,384$ (LDM, $f=4, 8$) |
| **Computational Complexity** | **{ddpm_flops:.2f} GFLOPs** (1000 sampling steps) | **{ldm_flops:.2f} GFLOPs** (**{flops_reduction:.1f}× lower FLOPs**) | LDM achieves $4\\times$ to $16\\times$ FLOPs reduction across ImageNet / CelebA (Rombach et al., 2022) |
| **Peak GPU VRAM Footprint** | **{ddpm_vram:.1f} MB** | **{ldm_vram:.1f} MB** (**{vram_reduction:.1f}× memory reduction**) | DDPM pixel training requires 32GB+ clusters; LDM trains on single consumer GPU (24GB VRAM) |
| **Training Throughput** | **{ddpm_epoch_sec:.2f} s / epoch** (Total: {ddpm_train_total_sec:.1f}s) | **{ldm_epoch_sec:.2f} s / epoch** (**{speed_speedup:.0f}% faster / {throughput_ratio:.2f}× throughput**); full pipeline {ldm_pipeline_total_sec:.1f}s including the {vae_train_total_sec:.1f}s VAE pre-training stage | LDM demonstrates ~2.5× faster wall-clock training per sample on 256×256 resolutions |
| **Sampling Latency (1000 steps)** | **{ddpm_sample_sec:.2f} s** ({ddpm_steps_per_sec:.1f} steps/s) | **{ldm_sample_sec:.2f} s** ({ldm_steps_per_sec:.1f} steps/s) | LDM enables DDIM 20–50 step acceleration without perceptual degradation |
| **Model Parameters** | U-Net: **{ddpm_params:,}** params | Latent U-Net: **{ldm_unet_params:,}** params + VAE: **{vae_params:,}** params (Total: **{ldm_total_params:,}**) | DDPM: 35.7M (CIFAR-10) / 114M (LSUN); LDM: 274M (LDM-4 on ImageNet) |
| **Optimization Objective Loss** | $L_\\text{{simple}} = {ddpm_loss:.4f}$ | $L_\\text{{LDM}} = {ldm_loss:.4f}$ (VAE MSE: {vae_mse:.4f}, PSNR: {vae_psnr:.1f} dB) | LDM loss reflects dense entropy in $4 \\times 7 \\times 7$ vs. sparse zero background in pixel space |
| **Synthesis Quality (FID / PSNR)** | Sample PSNR: **{ddpm_sample_psnr:.2f} dB**; Clean digit boundaries | Sample PSNR: **{ldm_sample_psnr:.2f} dB**; Slight boundary waviness from $f=4$ decode | DDPM CIFAR-10 FID: **3.17**; LDM ImageNet-256 FID: **3.60** (state-of-the-art competitive) |
| **Conditioning Mechanism** | Additive class embedding added to time-step feature map | **Cross-Attention Multi-Head Mechanism** $\\text{{Attn}}(Q=W_Q \\phi, K=W_K \\tau(y), V=W_V \\tau(y))$ | LDM supports arbitrary multimodal conditioning (text prompts, layout, depth, segmentation) |

> *Note on sources:* the first two columns are **our own measurements** recorded in `ddpm_pixel_stats.json`, `ldm_latent_stats.json` and `vae_stats.json` (MNIST, $N = 60,000$, 50 epochs, seed 42, single NVIDIA RTX PRO 6000 Blackwell). The last column contains values **cited from the original papers** (Ho et al., 2020; Rombach et al., 2022) — it is a literature reference, not a measurement of our runs.

## 2. Key Critical Insights (Section 4.2.4)

1. **Computational Advantage:**
   Operating in the $4 \\times 7 \\times 7$ latent manifold eliminates spatial redundancies, reducing inference FLOPs by **{flops_reduction:.1f}×** and peak VRAM by **{vram_reduction:.1f}×**, confirming Rombach et al.'s central hypothesis.
2. **Dense Latent Entropy vs. Pixel Sparsity:**
   The training loss of LDM ($L_\\text{{LDM}} \\approx {ldm_loss:.4f}$) is higher than pixel DDPM ($L_\\text{{simple}} \\approx {ddpm_loss:.4f}$). In direct pixel space, the majority of MNIST pixels are empty black background (zero variance), making noise prediction trivial for most elements. In contrast, VAE compresses image semantics into a dense, continuous $4 \\times 7 \\times 7$ tensor with high information entropy per coordinate.
3. **Downsampling Artifacts:**
   An aggressive downsampling factor of $f=4$ on a tiny $28 \\times 28$ image contracts spatial resolution to just $7 \\times 7$. The VAE reconstruction fidelity (PSNR $\\approx {vae_psnr:.2f}$ dB) bounds the quality achievable after decoding, so residual latent errors surface as subtle stroke-boundary waviness in the generated image.
4. **Two-Tier Cost Model and Latency-Bound Sampling:**
   The {speed_speedup:.0f}% per-epoch gain of LDM must be read together with its two-stage cost: training the VAE first consumes {vae_train_total_sec:.1f}s, so the end-to-end LDM pipeline takes {ldm_pipeline_total_sec:.1f}s against {ddpm_train_total_sec:.1f}s for pixel-space DDPM, i.e. only {pipeline_speedup:.2f}× overall. Sampling shows the same regime effect: although the latent U-Net needs {flops_reduction:.1f}× fewer FLOPs, the 1000-step trajectory takes {ldm_sample_sec:.2f}s versus {ddpm_sample_sec:.2f}s, because at $7 \\times 7$ resolution the GPU is dominated by kernel-launch latency rather than arithmetic, and the extra VAE decode adds one more stage to the critical path. Latent diffusion therefore buys *capacity*, not latency, at this scale — it must be combined with a reduced step count (e.g. DDIM) to realise a wall-clock win.
"""

    out_md = TABLES_DIR / "pixel_vs_latent_benchmark.md"
    out_md.write_text(md_content, encoding="utf-8")
    print(f"[Step 4: Benchmark] Generated markdown report: {out_md}")
    print("[Step 4: Benchmark] Completed successfully!")


if __name__ == "__main__":
    main()
