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

#figure(
  caption: [Progressive noise addition at five noise levels on a small image and pretrained denoising/generation results (Task 4).],
)[
  #fig-placeholder([
    Insert experimental image sequence (Task 4 & Section IV): \
    Row 1: Clean small image ($28 times 28$) + 5 increasing noise levels | Row 2: Pretrained denoising / generation example.
  ], height: 3.4cm)
] <fig:exp-results>

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
