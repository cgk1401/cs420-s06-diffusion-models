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

=== Experimental Setup and Forward Noising (Task 4a)

Following the experimental protocol established in Section 3.2, all empirical evaluations are conducted on the $28 times 28$ MNIST benchmark using PyTorch on a standard workstation CPU. The random seed is fixed to 42 to ensure exact reproducibility across all evaluations. The clean image $x_0$ retains the normalized intensity range of $[-1.0, 1.0]$, and the forward noise schedule adheres strictly to the linear parameters of Ho et al. @ho2020ddpm with $T = 1000$ timesteps, spanning from $beta_1 = 10^(-4)$ to $beta_T = 0.02$.

Applying the closed-form formulation derived in @eq:reparam, any intermediate noisy state $x_t$ can be evaluated directly from the uncorrupted input $x_0$ without simulating intervening Markov transitions. We sample this trajectory across five representative timesteps: $t in {50, 200, 400, 700, 1000}$ (corresponding to schedule indices $49, 199, 399, 699, 999$). The resulting image sequence, tracking the canonical digit from Section 3.2 alongside its highlighted foreground stroke pixel at coordinate $(10, 14)$, is presented in @fig:exp-results.

#figure(
  image("../../figures/fig_forward_5levels.pdf", width: 100%),
  caption: [Closed-form forward noising $q(x_t | x_0)$ at five progressive noise levels on a $28 times 28$ MNIST digit ($t in {50, 200, 400, 700, 1000}$, Seed = 42). The highlighted bounding box at coordinate $(10, 14)$ tracks the stroke pixel calculated in Section 3.2.],
) <fig:exp-results>

The visual degradation in @fig:exp-results directly reflects the analytical trade-off between the signal retention coefficient $sqrt(overline(alpha)_t)$ and the noise scale $sqrt(1 - overline(alpha)_t)$. At $t = 50$, signal retention remains dominant ($sqrt(overline(alpha)_(50)) = 0.99$), preserving the digit structure; at the highlighted coordinate $(10, 14)$, the pixel evaluates to $x_(50) = 0.72$, exactly corroborating the numerical extension in Section 3.2. As diffusion proceeds through $t = 200$ and $t = 400$, the injected noise gradually overtakes data variance ($sqrt(1 - overline(alpha)_(400)) = 0.72 > sqrt(overline(alpha)_(400)) = 0.69$), blurring boundaries and dissolving fine contours into the noise floor. By $t = 700$ and $t = 1000$, all semantic details vanish as $sqrt(overline(alpha)_(1000)) approx 0.00$, leaving an isotropic Gaussian state $x_(1000) ~ cal(N)(bold(0), bold(I))$.

This progression confirms that the forward process smoothly maps the complex data distribution $q(x_0)$ into a tractable prior $cal(N)(bold(0), bold(I))$. This provides the foundational boundary condition for generative modeling, enabling a neural network to synthesize new data by learning to reverse this trajectory from pure noise.

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
