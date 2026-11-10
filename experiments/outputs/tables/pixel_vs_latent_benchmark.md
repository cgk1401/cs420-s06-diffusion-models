\# Diffusion Models Benchmark: Pixel-Space (DDPM) vs. Latent-Space (LDM)

This benchmark provides an empirical, reproducible comparison between **Pixel-Space DDPM** (Ho et al., 2020) and **Latent Diffusion Models - LDM** (Rombach et al., 2022) conducted on the full MNIST dataset ($N = 60,000$, 50 epochs, seed 42) along with cited literature results.

## 1. Summary Comparison Table (Section 4.2.3 / Table 3)

| Explicit Criterion | Pixel-Space Diffusion (DDPM) | Latent Diffusion (LDM) | Literature Benchmark (Ho 2020 / Rombach 2022) |
| :--- | :--- | :--- | :--- |
| **Operating Space & Dimensions** | Direct pixel space: $1 \times 28 \times 28 = 784$ scalar pixels | Latent space: $4 \times 7 \times 7 = 196$ elements ($f=4$, $16\times$ area reduction via Spatial VAE) | ImageNet $256 \times 256 \times 3 = 196,608$ (DDPM) vs. $3 \times 64 \times 64 = 16,384$ (LDM, $f=4, 8$) |
| **Computational Complexity** | **117.87 GFLOPs** (1000 sampling steps) | **8.31 GFLOPs** (**14.2× lower FLOPs**) | LDM achieves $4\times$ to $16\times$ FLOPs reduction across ImageNet / CelebA (Rombach et al., 2022) |
| **Peak GPU VRAM Footprint** | **230.6 MB** | **37.6 MB** (**6.1× memory reduction**) | DDPM pixel training requires 32GB+ clusters; LDM trains on single consumer GPU (24GB VRAM) |
| **Training Throughput** | **9.42 s / epoch** (Total: 471.2s) | **5.74 s / epoch** (**39% faster / 1.64× throughput**); full pipeline 458.6s including the 171.7s VAE pre-training stage | LDM demonstrates ~2.5× faster wall-clock training per sample on 256×256 resolutions |
| **Sampling Latency (1000 steps)** | **1.59 s** (628.2 steps/s) | **1.71 s** (583.2 steps/s) | LDM enables DDIM 20–50 step acceleration without perceptual degradation |
| **Model Parameters** | U-Net: **179,809** params | Latent U-Net: **181,540** params + VAE: **99,785** params (Total: **281,325**) | DDPM: 35.7M (CIFAR-10) / 114M (LSUN); LDM: 274M (LDM-4 on ImageNet) |
| **Optimization Objective Loss** | $L_\text{simple} = 0.0210$ | $L_\text{LDM} = 0.1499$ (VAE MSE: 0.0010, PSNR: 30.1 dB) | LDM loss reflects dense entropy in $4 \times 7 \times 7$ vs. sparse zero background in pixel space |
| **Synthesis Quality (FID / PSNR)** | Sample PSNR: **8.09 dB**; Clean digit boundaries | Sample PSNR: **6.66 dB**; Slight boundary waviness from $f=4$ decode | DDPM CIFAR-10 FID: **3.17**; LDM ImageNet-256 FID: **3.60** (state-of-the-art competitive) |
| **Conditioning Mechanism** | Additive class embedding added to time-step feature map | **Cross-Attention Multi-Head Mechanism** $\text{Attn}(Q=W_Q \phi, K=W_K \tau(y), V=W_V \tau(y))$ | LDM supports arbitrary multimodal conditioning (text prompts, layout, depth, segmentation) |

> *Note on sources:* the first two columns are **our own measurements** recorded in `ddpm_pixel_stats.json`, `ldm_latent_stats.json` and `vae_stats.json` (MNIST, $N = 60,000$, 50 epochs, seed 42, single NVIDIA RTX PRO 6000 Blackwell). The last column contains values **cited from the original papers** (Ho et al., 2020; Rombach et al., 2022) — it is a literature reference, not a measurement of our runs.

## 2. Key Critical Insights (Section 4.2.4)

1. **Computational Advantage:**
   Operating in the $4 \times 7 \times 7$ latent manifold eliminates spatial redundancies, reducing inference FLOPs by **14.2×** and peak VRAM by **6.1×**, confirming Rombach et al.'s central hypothesis.
2. **Dense Latent Entropy vs. Pixel Sparsity:**
   The training loss of LDM ($L_\text{LDM} \approx 0.1499$) is higher than pixel DDPM ($L_\text{simple} \approx 0.0210$). In direct pixel space, the majority of MNIST pixels are empty black background (zero variance), making noise prediction trivial for most elements. In contrast, VAE compresses image semantics into a dense, continuous $4 \times 7 \times 7$ tensor with high information entropy per coordinate.
3. **Downsampling Artifacts:**
   An aggressive downsampling factor of $f=4$ on a tiny $28 \times 28$ image contracts spatial resolution to just $7 \times 7$. The VAE reconstruction fidelity (PSNR $\approx 30.10$ dB) bounds the quality achievable after decoding, so residual latent errors surface as subtle stroke-boundary waviness in the generated image.
4. **Two-Tier Cost Model and Latency-Bound Sampling:**
   The 39% per-epoch gain of LDM must be read together with its two-stage cost: training the VAE first consumes 171.7s, so the end-to-end LDM pipeline takes 458.6s against 471.2s for pixel-space DDPM, i.e. only 1.03× overall. Sampling shows the same regime effect: although the latent U-Net needs 14.2× fewer FLOPs, the 1000-step trajectory takes 1.71s versus 1.59s, because at $7 \times 7$ resolution the GPU is dominated by kernel-launch latency rather than arithmetic, and the extra VAE decode adds one more stage to the critical path. Latent diffusion therefore buys *capacity*, not latency, at this scale — it must be combined with a reduced step count (e.g. DDIM) to realise a wall-clock win.
