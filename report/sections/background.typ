#import "../helpers.typ": guideline, fig-placeholder

= Background and Paper Review

#guideline("Section 2 | Target: ~1.75–2.0 pages | Task 1 | Rubric: Reading & Synthesis (15 pts)")[
  Explain background concepts in your own words and review the two papers at a high level (keep detailed equations and architecture mechanics for Section 3).
]

== Discriminative vs. Generative Models

#guideline("Task 1a")[
  - Distinguish discriminative models learning $p(y | x)$ from generative models learning $p(x)$ or $p(x | y)$ to sample new data.
  - Give brief examples for each paradigm.
]

== Qualitative Comparison of Generative Models

#guideline("Task 1b")[
  - Qualitatively compare *GANs* @goodfellow2014gan, *Autoregressive Models*, and *Diffusion Models* @sohl2015deep @ho2020ddpm (optionally include *VAEs* @kingma2014vae as background for LDM).
  - Discuss trade-offs across sample quality, training stability, mode coverage, and sampling speed (@tab:gen-comparison).
]

#figure(
  caption: [Qualitative comparison of major generative modeling families (Task 1).],
)[
  #set text(size: 9.2pt)
  #table(
    columns: (1.3fr, 1.1fr, 1.2fr, 1.1fr, 1.3fr),
    align: (left, left, left, left, left),
    stroke: none,
    inset: (x: 6pt, y: 5pt),
    table.hline(stroke: 0.9pt),
    [*Criterion*], [*GANs*], [*Autoregressive*], [*VAEs*], [*Diffusion Models*],
    table.hline(stroke: 0.5pt),
    [Core Principle], [_\[Fill summary\]_], [_\[Fill summary\]_], [_\[Fill summary\]_], [_\[Fill summary\]_],
    [Sample Quality], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_],
    [Training Stability], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_],
    [Mode Coverage], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_],
    [Sampling Speed], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_], [_\[Fill\]_],
    table.hline(stroke: 0.9pt),
  )
] <tab:gen-comparison>

== Review of Core Papers: From DDPM to LDM

#guideline("Synthesis of Required & Recommended Papers")[
  - *Required Paper — DDPM* @ho2020ddpm — Objectives, main contributions (noise-prediction parameterization, simplified objective $L_"simple"$, connection to score matching), headline results, and pixel-space computational bottleneck.
  - *Recommended Paper — LDM* @rombach2022ldm — How LDM resolves DDPM's bottleneck by separating _perceptual compression_ (autoencoder) from _semantic generation_ (latent diffusion) and adding _cross-attention_ conditioning.
]
