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

All empirical evaluations are conducted on the full MNIST training set ($60,000$ samples) using PyTorch with CUDA GPU acceleration, fixing the global random seed to 42 for exact numerical reproducibility with Section 3.2. We train a conditional U-Net containing $179,809$ trainable parameters, equipped with 64-dimensional sinusoidal time embeddings, class embeddings for digits 0–9, and a cross-attention bottleneck. Optimization is performed using AdamW (initial learning rate $2 times 10^(-3)$, weight decay $10^(-4)$) coupled with a Cosine Annealing learning rate schedule decaying to $10^(-5)$ over 20 epochs with a batch size of 64. On a single GPU workstation, training completes in $198.8$ seconds (averaging $9.94$ seconds per epoch), with the simplified objective $L_"simple"$ converging monotonically to $0.0210$.

=== Forward Noising Across Five Noise Levels (Task 4a)

Following the closed-form forward sampling formulation in @eq:reparam, the noisy state $x_t$ is sampled directly across five representative timesteps: $t in {50, 200, 400, 700, 1000}$ (corresponding to schedule indices $49, 199, 399, 699, 999$). The resulting progression is presented in Row 1 of @fig:exp-results, where a red bounding box tracks the canonical foreground stroke coordinate $(10, 14)$ analyzed in Section 3.2.

#figure(
  image("../../figures/fig_exp_pixel_forward_reverse.pdf", width: 100%),
  caption: [Forward noising $q(x_t | x_0)$ (Row 1) and reverse denoising $p_theta(x_(t-1) | x_t)$ (Row 2) on MNIST ($T = 1000$). Red bounding boxes track pixel $(10, 14)$ from Section 3.2.],
) <fig:exp-results>

The visual degradation across Row 1 reflects the analytical trade-off between the signal retention scale $sqrt(overline(alpha)_t)$ and the noise scale $sqrt(1 - overline(alpha)_t)$. At $t = 50$, the signal strongly dominates ($sqrt(overline(alpha)_(50)) = 0.9877$, $sqrt(1 - overline(alpha)_(50)) = 0.1561$), preserving digit geometry while evaluating to $x_(50) = 0.7224$ at the tracked pixel, in exact agreement with the numerical derivation in Section 3.2. As diffusion advances through $t = 200$ and $t = 400$, noise variance progressively overtakes data variance ($sqrt(1 - overline(alpha)_(400)) = 0.7233 > sqrt(overline(alpha)_(400)) = 0.6905$), causing stroke boundaries to dissolve into the noise floor. By $t = 700$ and $t = 1000$, signal retention diminishes to near zero ($sqrt(overline(alpha)_(1000)) approx 0.0635$), yielding an isotropic Gaussian state $x_(1000) ~ cal(N)(bold(0), bold(I))$ and verifying complete information erasure.

=== Pixel-Space Reverse Denoising and Generation (Task 4b)

Unlike forward diffusion, generation requires sequential evaluation of the learned reverse transitions $p_theta(x_(t-1) | x_t)$ parameterized according to @eq:reverse and @eq:mu_param. Initializing from standard Gaussian noise $x_(1000) ~ cal(N)(bold(0), bold(I))$ conditioned on class label $y = 3$, ancestral sampling is executed across all $T = 1000$ discrete reverse steps, with intermediate states captured at $t in {1000, 700, 400, 200, 50, 0}$ displayed in Row 2 of @fig:exp-results.

The reverse trajectory exhibits a clear coarse-to-fine structural evolution. Starting from uncorrelated white noise at $t = 1000$, the model establishes low-frequency spatial priors by $t = 700$, concentrating image energy toward the canvas center. By $t = 400$, the dual-loop topological envelope of digit 3 emerges through the residual noise grain. Stroke geometry solidifies at $t = 200$ as background perturbations are suppressed toward zero intensity. Subsequent high-frequency refinement at $t = 50$ sharpens stroke boundaries, culminating at $t = 0$ in a clean, high-contrast, artifact-free synthesis of digit 3 on a solid dark background. Crucially, the tracked coordinate $(10, 14)$ aligns directly with the central stroke juncture of the generated sample, confirming that the reverse chain faithfully recovers the underlying empirical data distribution.

== Comparison: Pixel-Space vs. Latent Diffusion

#guideline("Task 6a & Section IV — Mandatory Evidence-Supported Comparison")[
  - Compare one pixel-space diffusion setting with one latent-diffusion setting using explicit criteria (@tab:pixel-vs-latent).
  - Discuss *sampling speed*, *generation quality* (citing FID/IS from @ho2020ddpm @rombach2022ldm), *compute/memory cost*, and *prompt-based control*.
]

#figure(
  caption: [Explicit comparison between pixel-space diffusion and latent-space diffusion (Task 6).],
)[
  #set text(size: 9.2pt)
  #table(
    columns: (1.3fr, 1.5fr, 1.5fr),
    align: (left, left, left),
    stroke: none,
    inset: (x: 6pt, y: 5pt),
    table.hline(stroke: 0.9pt),
    [*Explicit Criterion*], [*Pixel-Space Diffusion (DDPM)*], [*Latent Diffusion (LDM)*],
    table.hline(stroke: 0.5pt),
    [Operating Space & Dimensions], [_\[Fill pixel dimensions\]_], [_\[Fill latent dimensions\]_],
    [Compute & Memory Cost], [_\[Fill cited/measured cost\]_], [_\[Fill cited/measured cost\]_],
    [Sampling Speed], [_\[Fill comparison\]_], [_\[Fill comparison\]_],
    [Image Quality (FID)], [_\[Cite paper benchmark\]_], [_\[Cite paper benchmark\]_],
    [Prompt-Based Control], [_\[Fill comparison\]_], [_\[Fill Cross-Attention / CFG\]_],
    table.hline(stroke: 0.9pt),
  )
] <tab:pixel-vs-latent>

== Critical Analysis and Limitations

#guideline("Task 6b & Rubric: Critical Analysis and Limitations (10 pts)")[
  - *Technical Limitations:* Sequential multi-step sampling latency; VAE perceptual compression bottleneck at high downsampling factors (fine details/text).
  - *Data, Copyright, and Bias (Mandatory — Task 6):* Discuss issues regarding web-scraped *training data*, *copyright* / style memorization, and demographic/cultural *bias*.
  - *Scope of Team Experiments:* Note the illustrative scope of the small-scale experiments without overstating conclusions.
]
