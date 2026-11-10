= Results and Discussion
== Team Illustrative Experiments
=== Experimental Setup

We analyze saved runs on the 60,000-image MNIST training split @mnist. The run logs record PyTorch 2.11.0, CUDA 13.0, and one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. Both denoisers use the same compact U-Net design: base width 32, time-embedding dimension 64, two coordinate input channels, and additive class conditioning together with single-token cross-attention. They have different input/output channel counts and parameter totals. Each denoiser was trained for 50 epochs with batch size 64, seed 42, AdamW weight decay $10^(-4)$, and cosine learning-rate decay to $10^(-5)$. Initial learning rates differ: 0.002 for pixel diffusion and 0.0025 for latent diffusion. The VAE was trained for 30 epochs at learning rate 0.003. Appendix A records commands and measurement scope. A fixed seed aids repeatability but does not guarantee identical results across devices or software versions.

=== Forward Noising and Reverse Generation

The forward demonstration uses five mathematical timesteps: 50, 200, 400, 700, and 1000. A shared noise tensor is used in @eq:reparam, giving valid marginal samples coupled for visualization rather than a sampled Markov trajectory. The signal/noise coefficients are approximately $0.9854/0.1702$ at step 50 and $0.4418/0.8971$ at step 400. At step 1000 the signal coefficient is approximately 0.00635, leaving a small residual signal. For display only, intensities are mapped to $[0,1]$ and clipped; numerical diffusion states are not restricted to that interval.

#figure(
  image("../../figures/fig_exp_pixel_forward_reverse.pdf", width: 100%),
  caption: [Team-generated MNIST demonstration @mnist. Top: clean image and five forward noise levels. Bottom: an independent, class-3 ancestral generation, shown at states $1000,699,399,199,49,0$. Red boxes mark zero-based coordinate $(10,14)$; the rows do not show recovery of the same image. Source: `step2_train_ddpm.py`; snapshot labels corrected from the recorder's post-transition convention.],
) <fig:exp-results>

The reverse row in @fig:exp-results starts from independent Gaussian noise and uses 1,000 learned transitions. A recognizable digit emerges as sampling proceeds. The marked coordinate links the location to the worked example, not the identity of an original pixel. Intermediate states are labeled after their recorded transition: code index 699 produces mathematical state 699, whereas index 999 records the initial state before sampling.

== Comparison of Pixel-Space and Latent Diffusion
=== Reconstruction and Class Conditioning

The VAE reduces each image to 196 latent coordinates. On the single selected test image, deterministic reconstruction gives MSE $9.7833 times 10^(-4)$ and Peak Signal-to-Noise Ratio (PSNR) 30.10 dB, computed on intensities in $[0,1]$ as $10 log_10(1/"MSE")$. @fig:exp-ldm-latent shows that this example retains the digit's visible structure. This is not a held-out-set average and does not establish that compression is harmless for all images.

#figure(
  image("../../figures/fig_exp_ldm_latent_and_noise.pdf", width: 100%),
  caption: [Team-generated VAE reconstruction example on MNIST @mnist, from `step3_train_ldm.py`. The center panel is the channel mean of a $4 times 7 times 7$ latent, min-max normalized for display. Its $1/16$ area illustrates spatial reduction; scalar storage decreases by $4 times$.],
) <fig:exp-ldm-latent>

#figure(
  image("../../figures/fig_exp_ldm_cross_attn_conditioning.pdf", width: 100%),
  caption: [Team-generated class-conditioned samples from `step3_train_ldm.py` on MNIST @mnist. Conditions are integer class labels, not free-form text. Initial latent noise is shared, but subsequent reverse-step noise differs across samples. Both additive conditioning and single-token cross-attention are active.],
) <fig:exp-ldm-cond>

@fig:exp-ldm-cond shows outputs for labels 0, 3, and 7. These recognizable examples illustrate class-conditioned generation, but cannot isolate the effect of attention from additive conditioning or subsequent stochastic draws. Because $M=1$, no varying token-selection weights are learned in this demonstration. General text-prompt alignment and causal comparisons between conditioning mechanisms remain outside its scope.

=== Recorded Compute and Runtime

@tab:pixel-vs-latent separates measurements from the literature discussed in Section 2. The table reports the checked-in run statistics. Conditions are similar rather than identical, especially learning rates and cached latent preprocessing. Estimated arithmetic counts convolution and linear-layer multiply-accumulates (MACs), using two floating-point operations (FLOPs) per MAC and batch size one. Attention matrix products, softmax, normalization, activations, interpolation, sampler arithmetic, and VAE decoding are excluded.

#figure(
  kind: table,
  caption: [Team experiment records: `ddpm_pixel_stats.json`, `ldm_latent_stats.json`, and `vae_stats.json`. Values are single-run observations, not a matched quality or hardware benchmark. Times include only the workloads specified below.],
  {
    set text(size: 9pt)
    set par(justify: false, first-line-indent: 0pt)
    table(
      columns: (1.5fr, 1fr, 1.5fr), stroke: none, align: left, inset: 4pt,
      table.hline(stroke: 0.8pt), [Criterion], [Pixel diffusion], [Latent diffusion], table.hline(stroke: 0.4pt),
      [Operating tensor], [$1 times 28 times 28$], [$4 times 7 times 7$],
      [U-Net parameters], [179,809], [181,540],
      [Additional VAE parameters], [None], [99,785],
      [Counted FLOPs/U-Net evaluation], [117.87 million], [8.31 million],
      [Counted FLOPs/1,000 evaluations], [117.87 billion], [8.31 billion],
      [Mean denoiser epoch time], [9.42 s], [5.74 s],
      [Denoiser epoch-time sum], [471.2 s], [286.9 s],
      [VAE epoch-time sum], [None], [171.7 s],
      [Combined epoch-time sum], [471.2 s], [458.6 s; preprocessing excluded],
      [Recorded 1,000-step sampling time], [1.592 s], [1.715 s, including decode],
      [Final training noise MSE], [0.0210], [0.1499; different space],
      [Conditioning in both models], table.cell(colspan: 2)[Additive class embedding and single-token cross-attention],
      table.hline(stroke: 0.8pt),
    )
  },
) <tab:pixel-vs-latent>

#figure(
  image("../figures/fig_speed_flops_comparison.pdf", width: 100%),
  caption: [Authors' visualization of the same saved values as @tab:pixel-vs-latent. The arithmetic estimate covers convolution/linear layers only. Training bars sum timed epochs and exclude data preparation and latent-cache construction. Sampling is one recorded trajectory per setting.],
) <fig:benchmark-compute>

The latent U-Net has approximately $14.19 times$ fewer counted operations per step. Its recorded mean epoch time is 39.1% lower, equivalent to $1.64 times$ the epoch throughput. However, adding VAE epochs reduces the apparent advantage to $1.03 times$ for the sum of training-epoch times. Encoding the dataset and building the latent cache were not timed in this sum, so it is not an end-to-end comparison. The sampling observation is also mixed: latent generation is approximately 7.7% slower despite lower counted arithmetic. Small GPU workloads may be affected by launch overhead and memory traffic, and latent sampling includes decoding, but these explanations were not established by a kernel-level profile. The pixel timing also includes recording and transferring intermediate snapshots, unlike the latent timing; repeated measurements with matched instrumentation are needed for a controlled latency comparison.

The saved peak GPU allocations have been omitted from the comparison because the reset interval covers different stages, and evaluation-only runs can overwrite the memory field while retaining historical training times. The two noise losses also do not rank image quality: their input distributions, effective signal-to-noise ratios, and training configurations differ. Both targets are unit Gaussian noise, and black MNIST background becomes $-1$ after normalization. The loss gap does not measure entropy or prove the absence of underfitting.

== Critical Analysis and Limitations

The observations support a reduction in counted denoiser arithmetic for this particular representation and backbone. They do not establish a universal speedup or transferable runtime ratio. Larger images also increase autoencoder costs. Reusing a pretrained encoder may amortize those costs across tasks, but this was not measured here. Latent compression can discard detail and may constrain reconstruction; one favorable reconstruction cannot characterize that effect across the dataset @rombach2022ldm.

Both implementations use 1,000-step ancestral sampling, with fresh reverse noise and no extra final-step noise. DDIM provides an alternative speed-quality trade-off in either representation @song2021ddim. Our study uses one seed, one dataset, small models, final-epoch checkpoints, and one recorded timing per configuration. No confidence intervals, distributional image-quality scores, class-accuracy statistics, or diversity evaluations were computed. The generated-to-reference PSNR values in the raw logs compare unrelated images and are not used as quality evidence. Published CIFAR-10 and ImageNet FIDs describe their original experiments, not these MNIST runs.

Training-data limitations matter beyond compute. MNIST contains handwritten digits rather than natural scenes, so its examples cannot establish performance on photographs, languages, or complex prompts @mnist. Keras documents MNIST as copyrighted by Yann LeCun and Corinna Cortes and distributed under CC BY-SA 3.0 @mnistlicense; we cite the dataset and preserve attribution. This provenance does not imply that all generative-model training data has the same permissions.

The original LDM text-to-image experiment uses LAION-400M, while its other tasks use datasets such as ImageNet and Places @rombach2022ldm. Large image-text corpora may contain copyrighted or personal material and imbalanced representations. Later extraction research demonstrates that some diffusion systems can reproduce training images @carlini2023extracting; this is distinct from the original LDM experiments and was not tested here. Bias can also affect which appearances or associations a model generates. Dataset documentation, permission-aware curation, deduplication, and output audits can help address these concerns, but do not by themselves guarantee absence of memorization or bias. Our small visual demonstration does not measure either risk.
