
= Results and Discussion

== Team Illustrative Experiments

=== Experimental Setup and Implementation Details

All empirical evaluations are conducted on the full MNIST training set ($60,000$ samples) using PyTorch with CUDA GPU acceleration, fixing the global random seed to 42 for exact numerical reproducibility with Section 3.2. We train a conditional U-Net containing $179,809$ trainable parameters, equipped with 64-dimensional sinusoidal time embeddings, class embeddings for digits 0–9, and a cross-attention bottleneck. Optimization is performed using AdamW (initial learning rate $2 times 10^(-3)$, weight decay $10^(-4)$) coupled with a Cosine Annealing learning rate schedule decaying to $10^(-5)$ over 50 epochs with a batch size of 64. On a single GPU workstation, training completes in $471.2$ seconds (averaging $9.42$ seconds per epoch), with the simplified objective $L_"simple"$ decreasing steadily from $0.0493$ to $0.0210$, and the final-epoch model used for all reported sampling results.

=== Forward Noising and Reverse Denoising (Tasks 4a & 4b)

Following the closed-form forward formulation in @eq:reparam, noisy states $x_t$ are sampled directly at $t in {50, 200, 400, 700, 1000}$ (schedule indices $49, 199, 399, 699, 999$) and shown in Row 1 of @fig:exp-results. As $t$ increases, the signal scale $sqrt(overline(alpha)_t)$ decays while the noise scale $sqrt(1 - overline(alpha)_t)$ grows: at $t = 50$ the signal still dominates ($0.9854$ vs. $0.1702$), by $t = 400$ noise overtakes it ($0.8971 > 0.4418$), and at $t = 1000$ the state is essentially isotropic Gaussian noise ($sqrt(overline(alpha)_(1000)) approx 0.00635$), confirming complete information erasure.

#figure(
  image("../../figures/fig_exp_pixel_forward_reverse.pdf", width: 100%),
  caption: [Forward noising $q(x_t | x_0)$ (Row 1) and reverse denoising $p_theta(x_(t-1) | x_t)$ (Row 2) on MNIST ($T = 1000$). Red bounding boxes track pixel $(10, 14)$ from Section 3.2.],
) <fig:exp-results>

Generation inverts this process by sequentially evaluating the learned transitions parameterized by @eq:reverse and @eq:mu_param. Starting from $x_(1000) ~ cal(N)(bold(0), bold(I))$ conditioned on $y = 3$, ancestral sampling over all $T = 1000$ steps produces the trajectory in Row 2 of @fig:exp-results. The reverse chain follows a coarse-to-fine progression: low-frequency priors emerge by $t = 700$, the dual-loop structure of digit 3 appears by $t = 400$, stroke geometry consolidates at $t = 200$, and high-frequency details sharpen by $t = 50$ until a clean digit is obtained at $t = 0$. The tracked coordinate $(10, 14)$ remains consistent across both rows, linking the numerical example in Section 3.2 to the observed pixel evolution.

== Comparison: Pixel-Space vs. Latent Diffusion

=== Two-Stage Latent Diffusion Architecture

To overcome the prohibitive computational burden of diffusing directly in high-dimensional pixel space, Latent Diffusion Models (LDM) @rombach2022ldm decouple the generative process into two dedicated functional stages. The first stage performs perceptual compression with the Spatial Variational Autoencoder introduced in @sec:ldm: each $1 times 28 times 28$ input is mapped to a $4 times 7 times 7$ latent by the encoder of @eq:vae-enc and reconstructed by the decoder of @eq:vae-dec, yielding the $f = 4$ downsampling ($16 times$ area reduction) discussed there. The autoencoder is trained for 30 epochs with the reconstruction and KL objective described in that section, using $beta_"KL" = 5 times 10^(-4)$. On canonical validation samples, this stage attains a reconstruction Mean Squared Error of $0.0010$ and a Peak Signal-to-Noise Ratio (PSNR) of $30.10$ dB, indicating that the compressed representation preserves the perceptually relevant structure of the digit.

The second stage runs diffusion entirely within the $4 times 7 times 7$ latent manifold using the conditional U-Net objective defined in @eq:lldm. Instead of injecting the class label through element-wise addition, the latent U-Net integrates the multi-head Cross-Attention mechanism of @eq:crossattn, in which queries are projected from intermediate U-Net feature maps while keys and values are projected from the conditioning token embedding $tau_theta(y)$. This design allows the same backbone to route class information directly into the spatial feature hierarchy without altering the latent diffusion objective.

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

=== Empirical Benchmark: Pixel DDPM vs. Latent LDM

Both models were trained and profiled under identical conditions: the same MNIST training split, noise schedule, batch size, epoch budget, random seed and GPU. The only structural difference is that the latent model requires an additional first-stage Spatial VAE (30 epochs, $beta_"KL" = 5 times 10^(-4)$), frozen before the diffusion stage begins.

#figure(
  kind: table,
  caption: [Empirical comparison of pixel-space DDPM and latent-space LDM under identical conditions. The first two columns are our own measurements; the last column reports values cited from @ho2020ddpm and @rombach2022ldm and was not reproduced in this work.],
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
      [196,608 → 16,384 (ImageNet 256)],

      [FLOPs per step],
      [117.87 M],
      [8.31 M (14.19$times$ fewer)],
      [$4 times$ to $16 times$ reduction],

      [FLOPs per 1,000 steps],
      [117.87 G],
      [8.31 G],
      [—],

      [Peak VRAM],
      [230.64 MB],
      [37.61 MB (6.13$times$ lower)],
      [Cluster vs. single 24 GB GPU],

      [Time per epoch],
      [9.42 s],
      [5.74 s (1.64$times$)],
      [$2.5 times$ at $256 times 256$],

      [Total training],
      [471.2 s],
      [458.6 s (incl. 171.7 s VAE)],
      [—],

      [Sampling, 1,000 steps],
      [1.592 s (628.2 steps/s)],
      [1.715 s (583.2 steps/s)],
      [DDIM 20–50 steps],

      [Parameters],
      [179,809],
      [281,325 (incl. 99,785 VAE)],
      [35.7 M / 274 M],

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
      [FID 3.17 / 3.60],

      [Conditioning],
      [Additive embedding],
      [Cross-attention],
      [Text, layout, depth],

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

Quality must likewise be read at two levels. The VAE reconstructs the canonical digit at 30.10 dB, so the compression stage is not the bottleneck of the system, whereas the sample-level PSNR of 8.09 dB and 6.66 dB measures the pixel-aligned agreement of one generated image with one reference image and is too weak a proxy to rank the two models. For distributional quality we therefore cite the published FID values, 3.17 for DDPM on CIFAR-10 @ho2020ddpm and 3.60 for LDM-4 on ImageNet @rombach2022ldm, which were not reproduced here. Finally, the conditioning interfaces differ in kind: the pixel model adds the class embedding to the time-embedding path, while the latent model routes it through cross-attention, the mechanism shown in @fig:exp-ldm-cond to steer one shared noise towards three distinct digits. That interface is what allows @rombach2022ldm to condition on text and layout, although our implementation exercises class tokens only.

== Critical Analysis and Limitations

=== Technical Limitations and Scope of the Experiments

Generation in both models is strictly sequential, and the 1,000 reverse steps of @eq:reverse must each be evaluated on the output of the previous one. The price of the stable variational objective of @ho2020ddpm is paid in latency, and our measurements quantify the naive sampler at 1.592 s and 1.715 s per trajectory, showing that latent diffusion alone does not alleviate it at low resolution; the effective remedies are fewer sampling steps, distillation, or caching of intermediate predictions. The second limitation is the compression stage: with $f = 4$ the latent retains only $7 times 7$ resolution, so fine geometric detail is systematically smoothed and the decoder cannot restore the exact position of a quantised stroke, the documented failure mode of aggressive latent compression in @rombach2022ldm. Our implementation is further restricted to a single class token, so the multimodal claim of the original work remains untested here.

These limitations bound the conclusions. Our experiments use 28$times$28 images from a single dataset, a 50-epoch budget, one seed, one GPU and U-Nets of roughly 180,000 parameters, two orders of magnitude smaller than the published models, so only the ratios in @tab:pixel-vs-latent carry over to full-scale training. Each configuration was run once, so differences of a few percent, including the 1.03$times$ end-to-end advantage, lie within run-to-run noise; and no distributional metric was computed, since FID and IS require a feature extractor and sample count this setting does not justify.

=== Data, Copyright, and Bias

Our experiments are trained exclusively on MNIST, released under a Creative Commons Attribution-ShareAlike licence; no web-scraped corpus, photograph or third-party artwork enters the pipeline, so the copyright exposure of the present study is minimal. Its limitations are nonetheless real: the corpus is demographically narrow, produced by a predominantly adult American population whose handwriting conventions differ markedly from other cultures, and audits report a small fraction of incorrect labels. A model trained on it inherits a narrow prior over stroke shape and orientation, visible in our samples as clean but stylistically uniform digits, so any claim of generalisable generation quality must be restricted to the MNIST domain.

The literature motivating this report is exposed to risks our setting avoids. The high-resolution latent diffusion models of @rombach2022ldm and their successors are trained on web-scraped image-text collections of billions of pairs in which copyrighted artworks and identifiable persons appear unfiltered, which raises a copyright problem, since near-verbatim training images can be extracted from a trained generator; a style problem, since the manner of an identifiable artist can be reproduced without copying any image exactly, leaving attribution and consent unresolved; and a bias problem, since these corpora over-represent the English-speaking and affluent parts of the internet, an asymmetry that cross-attention can even steer deliberately. Countermeasures include deduplication and opt-out lists, provenance metadata such as C2PA, and auditing outputs for memorised content.
