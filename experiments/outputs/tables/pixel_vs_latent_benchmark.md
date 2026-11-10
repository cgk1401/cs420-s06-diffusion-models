# Recorded MNIST comparison

Source: checked-in ddpm_pixel_stats.json, ldm_latent_stats.json and vae_stats.json. Single-run descriptive observations; learning rates, preprocessing and sampling instrumentation differ.

| Metric | Pixel | Latent |
|---|---:|---:|
| Counted U-Net arithmetic (million FLOPs / evaluation) | 117.87 | 8.31 |
| Mean denoiser epoch (seconds / epoch) | 9.42 | 5.74 |
| Sum of timed training epochs (seconds; latent includes VAE) | 471.2 | 458.6 |
| Recorded sampling trajectory (seconds; instrumentation differs) | 1.592 | 1.715 |

Counted arithmetic ratio: 14.19x, using convolution/linear MACs only and 2 FLOPs/MAC. Other operators and VAE decode are excluded.

Training sums exclude data preparation and latent-cache construction. Pixel sampling records intermediate CPU snapshots; latent sampling includes decoding. These are not matched latency measurements.

Both models use additive class embeddings and single-token cross-attention. No spatial token-selection mechanism is established by the class examples. No distributional quality ranking is claimed.

Memory comparison is omitted: legacy peak-allocation fields do not identify standardized workloads and may have been refreshed during evaluation. Raw JSON is preserved unchanged.
