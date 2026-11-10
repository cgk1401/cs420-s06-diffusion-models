#import "../helpers.typ": guideline, fig-placeholder

= Results and Discussion

#guideline("Section 4 | Target: ~2.5–3.0 pages | Tasks 4, 6 & Section IV | Rubric: Evidence (10 pts) + Critical Analysis (10 pts)")[
  *Important (Section VI):* Clearly separate results from your team's own demonstrations (Section 4.1) and benchmark metrics cited from the original papers (Section 4.2).
]

== Team Illustrative Experiments

#guideline("Task 4 & Section IV (Minimum Expected Outputs)")[
  - *Setup:* Describe environment, small image dataset (e.g., $28 times 28$ MNIST), noise schedule, and pretrained model used (no training from scratch required).
  - *Experiment 1 — Noise Addition at Five Noise Levels (Task 4a — Mandatory):* Show clean small image $x_0$ and five progressive noise levels (@fig:exp-results).
  - *Experiment 2 — Denoising / Generation with Pretrained Model (Task 4b):* Demonstrate reverse denoising or sample generation if resources permit.
  - *Experiment 3 — Pixel vs. Latent Compression Check:* Optional lightweight check comparing pixel tensor vs. VAE latent tensor dimensions and reconstruction quality.
]

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

#guideline("Task 6a & Section IV — Mandatory Evidence-Supported Comparison")[
  - Compare one pixel-space diffusion setting with one latent-diffusion setting using explicit criteria.
  - Discuss *sampling speed*, *generation quality* (citing FID/IS from @ho2020ddpm @rombach2022ldm), *compute/memory cost*, and *prompt-based control*.
]

=== Two-Stage Latent Diffusion Architecture

To overcome the prohibitive computational burden of diffusing directly in high-dimensional pixel space, Latent Diffusion Models (LDM) @rombach2022ldm decouple the generative process into two dedicated functional stages. The first stage performs perceptual compression with the Spatial Variational Autoencoder introduced in @sec:ldm: each $1 times 28 times 28$ input is mapped to a $4 times 7 times 7$ latent by the encoder of @eq:vae-enc and reconstructed by the decoder of @eq:vae-dec, yielding the $f = 4$ downsampling ($16 times$ area reduction) discussed there. The autoencoder is trained for 30 epochs with the reconstruction and KL objective described in that section, using $beta_"KL" = 5 times 10^(-4)$. On canonical validation samples, this stage attains a reconstruction Mean Squared Error of $0.0010$ and a Peak Signal-to-Noise Ratio (PSNR) of $29.86$ dB, indicating that the compressed representation preserves the perceptually relevant structure of the digit.

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

// TODO: viết lại sau khi đọc code và chạy thực nghiệm.

=== Detailed Comparative Discussion

// TODO: viết lại sau khi đọc code và chạy thực nghiệm.

== Critical Analysis and Limitations

#guideline("Task 6b & Rubric: Critical Analysis and Limitations (10 pts)")[
  - *Technical Limitations:* Sequential multi-step sampling latency; VAE perceptual compression bottleneck at high downsampling factors (fine details/text).
  - *Data, Copyright, and Bias (Mandatory — Task 6):* Discuss issues regarding web-scraped *training data*, *copyright* / style memorization, and demographic/cultural *bias*.
  - *Scope of Team Experiments:* Note the illustrative scope of the small-scale experiments without overstating conclusions.
]

// TODO: viết lại sau khi đọc code và chạy thực nghiệm.
