
= Results and Discussion

== Team Illustrative Experiments

=== Experimental Setup and Implementation Details

All empirical evaluations are conducted on the full MNIST training set ($60,000$ samples) using PyTorch 2.11.0 with CUDA 13.0 on a single NVIDIA RTX PRO 6000 Blackwell Server Edition GPU, fixing the global random seed to 42 for exact numerical reproducibility with Section 3.2. We train a conditional U-Net containing $179,809$ trainable parameters, equipped with 64-dimensional sinusoidal time embeddings, class embeddings for digits 0–9, and a cross-attention bottleneck. Optimization is performed using AdamW, the Adam optimizer with decoupled weight decay (initial learning rate $2 times 10^(-3)$, weight decay $10^(-4)$) coupled with a Cosine Annealing learning rate schedule decaying to $10^(-5)$ over 50 epochs with a batch size of 64. On this GPU, training completes in $471.2$ seconds (averaging $9.42$ seconds per epoch), with the simplified objective $L_"simple"$ decreasing steadily from $0.0493$ to $0.0210$, and the final-epoch model used for all reported sampling results.

=== Forward Noising and Reverse Denoising (Tasks 4a and 4b)

Following the closed-form forward formulation in @eq:reparam, noisy states $x_t$ are sampled directly at $t in {50, 200, 400, 700, 1000}$ (schedule indices $49, 199, 399, 699, 999$) and shown in Row 1 of @fig:exp-results. As $t$ increases, the signal scale $sqrt(overline(alpha)_t)$ decays while the noise scale $sqrt(1 - overline(alpha)_t)$ grows: at $t = 50$ the signal still dominates ($0.9854 > 0.1702$), by $t = 400$ noise overtakes it ($0.8971 > 0.4418$), and at $t = 1000$ the state is essentially isotropic Gaussian noise ($sqrt(overline(alpha)_(1000)) approx 0.00635$), confirming complete information erasure.

#figure(
  image("../../figures/fig_exp_pixel_forward_reverse.pdf", width: 100%),
  caption: [Forward noising $q(x_t | x_0)$ (Row 1) and reverse denoising $p_theta(x_(t-1) | x_t)$ (Row 2) on MNIST ($T = 1000$). Red bounding boxes track pixel $(10, 14)$ from Section 3.2.],
) <fig:exp-results>

Generation inverts this process by sequentially evaluating the learned transitions parameterized by @eq:reverse and @eq:mu_param. Starting from $x_(1000) ~ cal(N)(bold(0), bold(I))$ conditioned on $y = 3$, ancestral sampling over all $T = 1000$ steps produces the trajectory in Row 2 of @fig:exp-results. The reverse chain follows a coarse-to-fine progression: low-frequency priors emerge by $t = 700$, the dual-loop structure of digit 3 appears by $t = 400$, stroke geometry consolidates at $t = 200$, and high-frequency details sharpen by $t = 50$ until a clean digit is obtained at $t = 0$. The tracked coordinate $(10, 14)$ remains consistent across both rows, linking the numerical example in Section 3.2 to the observed pixel evolution.

== Comparison of Pixel-Space and Latent Diffusion

=== Two-Stage Latent Diffusion Architecture

To overcome the prohibitive computational burden of diffusing directly in high-dimensional pixel space, LDM @rombach2022ldm decouples the generative process into two dedicated functional stages. The first stage performs perceptual compression with the spatial VAE introduced in @sec:ldm: each $1 times 28 times 28$ input is mapped to a $4 times 7 times 7$ latent by the encoder of @eq:vae-enc and reconstructed by the decoder of @eq:vae-dec, yielding the $f = 4$ downsampling ($16 times$ area reduction) discussed there. The autoencoder is trained for 30 epochs with the reconstruction and KL objective of @eq:lvae, using $beta_"KL" = 5 times 10^(-4)$. On canonical validation samples, this stage attains a reconstruction MSE of $9.78 times 10^(-4)$ and a Peak Signal-to-Noise Ratio (PSNR) of $30.10$ dB, indicating that the compressed representation preserves the perceptually relevant structure of the digit.

The second stage runs diffusion entirely within the $4 times 7 times 7$ latent manifold using the conditional U-Net objective defined in @eq:lldm. The latent denoiser is the same conditional U-Net used for the pixel-space model in the previous subsection, with the same conditioning mechanism: the class embedding $tau_theta(y)$ is added to the time embedding that modulates every residual block, and the same embedding is also injected through the Cross-Attention mechanism of @eq:crossattn at two locations (after the first encoder block and at the bottleneck), in which queries are projected from intermediate U-Net feature maps while keys and values are projected from $tau_theta(y)$. The only differences from the pixel model are the operating space and the number of input channels (four latent channels instead of one image channel).

The compression and reconstruction flow of the first stage is visualized in @fig:exp-ldm-latent.

#figure(
  align(center, image("../../figures/fig_exp_ldm_latent_and_noise.pdf", width: 70%)),
  caption: [VAE compression and reconstruction on MNIST: input $x_0$ ($28 times 28$), latent $z_0$ ($7 times 7$, shown at $1/16$ area), and decoded $tilde(x)$ ($28 times 28$).],
) <fig:exp-ldm-latent>

=== Prompt Steering via Cross-Attention Conditioning

A foundational capability of LDM is steering an identical stochastic initial state into semantically distinct outputs through Cross-Attention routing. To provide direct visual evidence of this mechanism, we initialize a single shared latent Gaussian noise realization $z_(1000) ~ cal(N)(bold(0), bold(I))$ of shape $4 times 7 times 7$ and execute three independent reverse trajectories conditioned on class prompts $y in {"'0'", "'3'", "'7'"}$.

#figure(
  align(center, image("../../figures/fig_exp_ldm_cross_attn_conditioning.pdf", width: 75%)),
  caption: [Cross-Attention conditioning from a shared latent noise $z_(1000)$ ($4 times 7 times 7$): the same initial state is steered toward $y = 0, 3, 7$, yielding distinct $28 times 28$ digits.],
) <fig:exp-ldm-cond>

As demonstrated in @fig:exp-ldm-cond, the shared initial noise cloud provides the spatial entropy base, while the Cross-Attention key-value projections dynamically modulate intermediate feature maps at each reverse timestep. Under prompt $y = 0$, attention weights guide the latent trajectory to form a single continuous elliptical loop; under $y = 3$, the model synthesizes two open curved lobes with a central cusp; under $y = 7$, probability mass concentrates into a prominent horizontal bar and descending diagonal stem. This demonstrates that LDM conditioning does not simply filter post-hoc samples, but deterministically dictates semantic trajectories from the very first denoising steps.

=== Empirical Benchmark of Pixel-Space DDPM and LDM

Both models were trained and profiled under matched conditions: the same MNIST training split, noise schedule, batch size, epoch budget (50 epochs), optimizer (AdamW, weight decay $10^(-4)$, cosine schedule to $10^(-5)$), random seed and GPU. Both diffusion stages use the same conditional U-Net architecture with the same conditioning mechanism; architecturally they differ only in the operating space and the number of input channels. Two differences remain and are stated explicitly: the latent U-Net uses an initial learning rate of $2.5 times 10^(-3)$ instead of $2 times 10^(-3)$, and because its input and output convolutions handle four latent channels instead of one image channel it has $181,540$ trainable parameters instead of $179,809$. The only additional component is that the latent model requires a first-stage spatial VAE (30 epochs, $beta_"KL" = 5 times 10^(-4)$), frozen before the diffusion stage begins.

@tab:pixel-vs-latent reports compute as floating-point operations (FLOPs) per U-Net forward pass, memory as peak video memory (VRAM) allocated on the GPU, and wall-clock time for training and for a full 1,000-step reverse trajectory. For the cited models, the sampling row lists the number of steps of the Denoising Diffusion Implicit Models (DDIM) sampler used by @rombach2022ldm, a deterministic sampler that skips intermediate timesteps.

#figure(
  kind: table,
  caption: [Empirical comparison of pixel-space DDPM and latent-space LDM under matched training conditions. The first two columns are our own measurements; the last column reports values cited from @ho2020ddpm and @rombach2022ldm and was not reproduced in this work.],
  {
    set text(size: 8.4pt)
    set par(justify: false, leading: 0.55em, spacing: 0.35em, first-line-indent: 0pt)
    table(
      columns: (1.1fr, 1.35fr, 1.6fr, 1.25fr),
      align: (left, left, left, left),
      stroke: none,
      inset: (x: 4pt, y: 3.5pt),

      table.hline(stroke: 0.9pt),
      [Criterion], [Pixel DDPM], [Latent LDM], [Literature (cited)],
      table.hline(stroke: 0.5pt),

      [Operating space],
      [$1 times 28 times 28 = 784$],
      [$4 times 7 times 7 = 196$ ($f = 4$)],
      [$256^2 times 3 = 196,608$ → $64^2 times 3 = 12,288$ (LDM-4, ImageNet)],

      [FLOPs per step],
      [117.9 M],
      [8.3 M (14.19$times$ fewer)],
      [—],

      [FLOPs per 1,000 steps],
      [117.9 G],
      [8.3 G],
      [—],

      [Peak VRAM],
      [230.6 MB],
      [37.6 MB (6.13$times$ lower)],
      [—],

      [Time per epoch],
      [9.42 s],
      [5.74 s (1.64$times$)],
      [—],

      [Total training],
      [471.2 s],
      [458.6 s (incl. 171.7 s VAE)],
      [—],

      [Sampling, 1,000 steps],
      [1.592 s (628.2 steps/s)],
      [1.715 s (583.2 steps/s)],
      [1,000 (DDPM) / 250 DDIM (LDM-4-G)],

      [Parameters],
      [179,809],
      [281,325 (181,540 U-Net + 99,785 VAE)],
      [35.7 M (DDPM, CIFAR-10) / 400 M (LDM-4-G)],

      [Final loss],
      [$L_"simple"$ = 0.0210],
      [$L_"LDM"$ = 0.1499],
      [—],

      [VAE reconstruction],
      [—],
      [MSE $9.78 times 10^(-4)$, PSNR 30.10 dB],
      [—],

      [Sample PSNR],
      [8.09 dB],
      [6.66 dB],
      [FID 3.17 (DDPM, CIFAR-10) / 3.60 (LDM-4-G, ImageNet)],

      [Conditioning],
      [Additive + cross-attention (shared U-Net)],
      [Additive + cross-attention (shared U-Net)],
      [Text, layout, semantic maps],

      table.hline(stroke: 0.9pt),
    )
  }
) <tab:pixel-vs-latent>

#figure(
  image("../../figures/fig_speed_flops_comparison.pdf", width: 100%),
  caption: [Compute, memory and training-throughput comparison between pixel-space DDPM and latent-space LDM on MNIST.],
) <fig:benchmark-compute>

Latent diffusion buys arithmetic, not latency. Compression reduces the operating space by a factor of four in scalar count and by sixteen in area, and the U-Net built on it requires 14.19$times$ fewer FLOPs per step and 6.13$times$ less peak memory. The gain is smaller than the area reduction because the latent U-Net consumes four input channels instead of one and shares the channel-width schedule of the pixel backbone.

This advantage must be read against the cost of the first stage. Per epoch the latent model is 39% faster, at 5.74 s against 9.42 s, but once the 171.7 s VAE stage is included the end-to-end cost is 458.6 s against 471.2 s, that is 1.03$times$. At $28 times 28$ the pixel-space baseline is cheap enough for an auxiliary 30-epoch run to absorb a third of the budget, whereas at $256 times 256$ the pixel-space cost grows by roughly two orders of magnitude while the VAE stage does not, and the amortisation becomes decisive. The same regime effect appears at sampling time: despite 14.19$times$ fewer FLOPs, the latent trajectory takes 1.715 s against 1.592 s, because at $7 times 7$ the workload is bound by kernel-launch and memory-access latency rather than by arithmetic, and the VAE decode adds one further stage to the critical path. A wall-clock gain therefore requires fewer reverse steps, such as DDIM sampling, rather than a smaller tensor alone.

The two objective values are not comparable across spaces. $L_"simple"$ = 0.0210 against $L_"LDM"$ = 0.1499 reflects that the great majority of MNIST coordinates are exactly zero, so most of the pixel-space target is trivial to predict, whereas the 196 latent coefficients carry the whole digit and thus have a higher variance. The higher latent loss is evidence for the entropy-packing argument of @rombach2022ldm, not a sign of underfitting.

Quality must likewise be read at two levels. The VAE reconstructs the canonical digit at 30.10 dB, so the compression stage is not the bottleneck of the system, whereas the sample-level PSNR of 8.09 dB and 6.66 dB measures the pixel-aligned agreement of one generated image with one reference image and is too weak a proxy to rank the two models. For distributional quality we therefore cite the published FID values, 3.17 for DDPM on CIFAR-10 @ho2020ddpm and 3.60 for the class-conditional LDM-4-G (400 M parameters, classifier-free guidance) on ImageNet $256 times 256$ @rombach2022ldm, which were not reproduced here. Finally, conditioning is not a point of difference in our setup: both models share one U-Net in which the class embedding is added to the time-embedding path and is simultaneously injected through two cross-attention blocks, the mechanism shown in @fig:exp-ldm-cond to steer one shared noise towards three distinct digits. The measured gaps in @tab:pixel-vs-latent therefore stem from the operating space and the number of input channels, not from the conditioning interface. Cross-attention is what allows @rombach2022ldm to condition on text and layout, although our implementation exercises class tokens only.

== Critical Analysis and Limitations

=== Technical Limitations and Scope of the Experiments

Generation in both models is strictly sequential, and the 1,000 reverse steps of @eq:reverse must each be evaluated on the output of the previous one. The price of the stable variational objective of @ho2020ddpm is paid in latency, and our measurements quantify the naive sampler at 1.592 s and 1.715 s per trajectory, showing that latent diffusion alone does not alleviate it at low resolution; the effective remedies are fewer sampling steps, distillation, or caching of intermediate predictions. The second limitation is the compression stage: with $f = 4$ the latent retains only $7 times 7$ resolution, so fine geometric detail is systematically smoothed and the decoder cannot restore the exact position of a quantised stroke, the documented failure mode of aggressive latent compression in @rombach2022ldm. Our implementation is further restricted to a single class token, so the multimodal claim of the original work remains untested here.

These limitations bound the conclusions. Our experiments use 28$times$28 images from a single dataset, a 50-epoch budget, one seed, one GPU and U-Nets of roughly 180,000 parameters, two orders of magnitude smaller than the published models, so only the ratios in @tab:pixel-vs-latent carry over to full-scale training. Each configuration was run once, so differences of a few percent, including the 1.03$times$ end-to-end advantage, lie within run-to-run noise; and no distributional metric was computed, since FID and IS require a feature extractor and sample count this setting does not justify.

=== Data, Copyright, and Bias

Our experiments are trained exclusively on MNIST, released under a Creative Commons Attribution-ShareAlike licence; no web-scraped corpus, photograph or third-party artwork enters the pipeline, so the copyright exposure of the present study is minimal. Its limitations are nonetheless real: the corpus is demographically narrow, produced by a predominantly adult American population whose handwriting conventions differ markedly from other cultures, and audits report a small fraction of incorrect labels. A model trained on it inherits a narrow prior over stroke shape and orientation, visible in our samples as clean but stylistically uniform digits, so any claim of generalisable generation quality must be restricted to the MNIST domain.

The literature motivating this report is exposed to risks our setting avoids. The high-resolution latent diffusion models of @rombach2022ldm and their successors are trained on web-scraped image-text collections of billions of pairs in which copyrighted artworks and identifiable persons appear unfiltered, which raises a copyright problem, since near-verbatim training images can be extracted from a trained generator; a style problem, since the manner of an identifiable artist can be reproduced without copying any image exactly, leaving attribution and consent unresolved; and a bias problem, since these corpora over-represent the English-speaking and affluent parts of the internet, an asymmetry that cross-attention can even steer deliberately. Countermeasures include deduplication and opt-out lists, provenance metadata such as the Coalition for Content Provenance and Authenticity (C2PA) standard, and auditing outputs for memorised content.
