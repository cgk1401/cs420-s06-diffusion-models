// ============================================================================
// CS420 SEMINAR RESEARCH REPORT TEMPLATE [S06] — SINGLE-COLUMN ACADEMIC PAPER
// Compatible with typst.app (multi-user real-time collaboration) & Typst CLI
// ============================================================================

// Toggle to `false` before final submission to hide all writing notes
// and verify the 6–10 A4 page limit (excluding References & Appendix).
#let show-guidelines = true

// ============================================================================
// DOCUMENT & ACADEMIC PAPER STYLING (SINGLE-COLUMN A4 RESEARCH FORMAT)
// ============================================================================
#set document(
  title: "Diffusion Models: From Adding Noise to Generating Images",
  author: ("Author Name 1", "Author Name 2", "Author Name 3"),
)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.2cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8.5pt, fill: luma(110))
      CS420 Seminar Research Report — Topic [S06]: Diffusion Models
      #h(1fr)
      _From Adding Noise to Generating Images_
      #v(-5pt)
      #line(length: 100%, stroke: 0.4pt + luma(190))
    ]
  },
)

#set text(
  font: ("New Computer Modern", "Linux Libertine", "Times New Roman"),
  size: 10.5pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.62em,
  spacing: 0.85em,
  first-line-indent: 1.2em,
)

#set heading(numbering: "1.1")

#show heading.where(level: 1): it => block(
  above: 1.45em,
  below: 0.7em,
  text(size: 12.5pt, weight: "bold", it),
)

#show heading.where(level: 2): it => block(
  above: 1.15em,
  below: 0.55em,
  text(size: 11pt, weight: "bold", it),
)

#set math.equation(numbering: "(1)", supplement: [Eq.])
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(size: 9.2pt)

// ============================================================================
// HELPER MACROS FOR WRITING GUIDELINES & FIGURE PLACEHOLDERS
// ============================================================================
#let guideline(tag, body) = {
  if show-guidelines {
    block(
      width: 100%,
      fill: rgb("#f2f6fc"),
      stroke: (left: 2.5pt + rgb("#2b579a"), rest: 0.4pt + rgb("#c5d5ea")),
      inset: 9pt,
      radius: 2pt,
      above: 0.7em,
      below: 0.85em,
    )[
      #set par(first-line-indent: 0pt, leading: 0.55em, spacing: 0.5em)
      #set text(size: 9.2pt, fill: luma(35))
      #text(weight: "bold", fill: rgb("#1e4278"))[\[WRITING GUIDE — #tag\]]
      #v(2pt)
      #body
    ]
  }
}

#let fig-placeholder(height: 3.4cm, label-text) = {
  rect(
    width: 100%,
    height: height,
    fill: luma(246),
    stroke: (paint: luma(160), thickness: 0.7pt, dash: "dashed"),
    radius: 3pt,
    inset: 12pt,
  )[
    #set par(first-line-indent: 0pt)
    #align(center + horizon)[
      #text(size: 9.2pt, fill: luma(85))[
        *\[FIGURE / DIAGRAM PLACEHOLDER\]* \
        #v(3pt)
        #label-text
      ]
    ]
  ]
}

// ============================================================================
// TITLE, AUTHORS & ABSTRACT BLOCK
// ============================================================================
#block(width: 100%)[
  #set par(first-line-indent: 0pt)
  #align(center)[
    #text(size: 9pt, weight: "bold", tracking: 0.5pt, fill: luma(90))[
      CS420: INTRODUCTION TO ARTIFICIAL INTELLIGENCE — SEMINAR RESEARCH REPORT [S06]
    ]
    #v(7pt)
    #text(size: 17pt, weight: "bold")[
      Diffusion Models: From Adding Noise to Generating Images
    ]
    #v(4pt)
    #text(size: 11.2pt, style: "italic", fill: luma(60))[
      From Pixel-Space Denoising Diffusion to High-Resolution Latent Diffusion
    ]
    #v(14pt)

    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 12pt,
      [
        *Author Name 1* \
        #text(size: 9pt, fill: luma(80))[Student ID: XXXXXXXX]
      ],
      [
        *Author Name 2* \
        #text(size: 9pt, fill: luma(80))[Student ID: XXXXXXXX]
      ],
      [
        *Author Name 3* \
        #text(size: 9pt, fill: luma(80))[Student ID: XXXXXXXX]
      ],
    )
    #v(6pt)
    #text(size: 9.2pt)[
      Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam \
      `{author1, author2, author3}@student.hcmus.edu.vn`
    ]
  ]

  #v(14pt)

  // Academic Abstract Block
  #pad(x: 1.0cm)[
    #align(center)[#text(size: 10.2pt, weight: "bold")[Abstract]]
    #v(4pt)
    #set text(size: 9.5pt)
    #guideline("Abstract | ~150–200 words")[
      - *Context:* Briefly introduce generative AI in image synthesis and the emergence of diffusion models.
      - *Core Question & Scope:* State how iterative Gaussian denoising enables novel image generation in DDPM @ho2020ddpm, and how Latent Diffusion Models (LDM) @rombach2022ldm reduce computational demands.
      - *Experiments:* Mention the 1D worked numerical example, 5-level forward noising / reverse denoising demonstrations, and pixel-vs.-latent comparison.
      - *Main Takeaway:* Summarize key findings regarding sampling speed, quality, prompt control, and limitations.
    ]
    #v(4pt)
    #text(size: 9.2pt)[
      *Index Terms*—Generative AI, Denoising Diffusion Probabilistic Models (DDPM), Latent Diffusion Models (LDM), Variational Autoencoder, Cross-Attention.
    ]
  ]

  #v(10pt)
  #line(length: 100%, stroke: 0.5pt + luma(180))
]

// ============================================================================
// 1. INTRODUCTION
// ============================================================================
= Introduction

#guideline("Section 1 | Target: ~0.75–1.0 page | Rubric: Problem Statement & Objectives (5 pts)")[
  - *Context:* Overview of modern AI image synthesis and prior challenges (training instability or high computational cost).
  - *Guiding Question (Mandatory — Section I):* Explicitly state the central question: _"How can learning to remove noise enable a model to generate new images?"_ and why latent diffusion reduces computational demands.
  - *Objectives & Scope:* Briefly introduce the two studied papers—DDPM @ho2020ddpm (Required) and LDM @rombach2022ldm (Recommended)—and state the report's objectives.
  - *Report Outline:* Summarize the organization of Sections 2–5.
]

// ============================================================================
// 2. BACKGROUND / PAPER REVIEW
// ============================================================================
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
  - *Required Paper — DDPM* @ho2020ddpm — Objectives, main contributions (noise-prediction parameterization, simplified objective $L_("simple")$, connection to score matching), headline results, and pixel-space computational bottleneck.
  - *Recommended Paper — LDM* @rombach2022ldm — How LDM resolves DDPM's bottleneck by separating _perceptual compression_ (autoencoder) from _semantic generation_ (latent diffusion) and adding _cross-attention_ conditioning.
]

// ============================================================================
// 3. METHOD / WORKED EXAMPLE
// ============================================================================
= Method and Worked Example

#guideline("Section 3 | Target: ~2.5–3.0 pages | Tasks 2, 3, 5 & Section III | Rubric: Algorithms & Equations (15 pts)")[
  Present the mathematical framework, symbol definitions, modeling assumptions, bidirectional diagrams, step-by-step numerical calculation, and LDM pipeline.
]

== DDPM: Forward Noising and Reverse Denoising

#guideline("Task 2 & Section III (Equations to Explain)")[
  - *Forward Process & $beta$ Schedule:* Explain progressive Gaussian noise injection via $beta_1, dots, beta_T$ (@eq:forward) and closed-form sampling $q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) bold(I))$.
  - *Reverse Process, Noise Predictor & Sampling:* Explain $p_theta (x_(t-1) | x_t)$, the U-Net noise predictor $epsilon_theta (x_t, t)$, the training loss (@eq:lsimple), and iterative sampling from $x_T tilde.op cal(N)(bold(0), bold(I))$ back to $x_0$.
  - *Mandatory Specification (Section III):* Define *every symbol*, specify units/value ranges, and state modeling assumptions (@tab:symbols).
]

$ q(x_t | x_(t-1)) = cal(N)(sqrt(1 - beta_t) x_(t-1), beta_t bold(I)) $ <eq:forward>

$ L_("simple") = bb(E)_(t, x_0, epsilon) [ norm(epsilon - epsilon_theta (x_t, t))^2 ] $ <eq:lsimple>

#figure(
  caption: [Symbol definitions, value ranges, and roles in @eq:forward and @eq:lsimple (Section III).],
)[
  #set text(size: 9.2pt)
  #table(
    columns: (0.9fr, 2.2fr, 1.3fr),
    align: (left, left, left),
    stroke: none,
    inset: (x: 6pt, y: 5pt),
    table.hline(stroke: 0.9pt),
    [*Symbol*], [*Definition & Role*], [*Range / Units*],
    table.hline(stroke: 0.5pt),
    [$x_0$], [_\[Define clean normalized data\]_], [_\[e.g., $[-1, 1]^D$\]_],
    [$x_t, x_(t-1)$], [_\[Define noisy states at $t, t-1$\]_], [_\[e.g., $bb(R)^D$\]_],
    [$t, T$], [_\[Define discrete timestep & total steps\]_], [_\[e.g., $t in {1, dots, T}$\]_],
    [$beta_t$], [_\[Define variance schedule at step $t$\]_], [_\[e.g., $beta_t in (0, 1)$\]_],
    [$bold(I)$], [_\[Define identity covariance matrix\]_], [_\[$bb(R)^(D times D)$\]_],
    [$epsilon$], [_\[Define standard Gaussian noise\]_], [_\[$epsilon tilde.op cal(N)(bold(0), bold(I))$\]_],
    [$epsilon_theta (x_t, t)$], [_\[Define neural noise predictor\]_], [_\[$bb(R)^D$\]_],
    table.hline(stroke: 0.9pt),
  )
] <tab:symbols>

#figure(
  caption: [Illustration of the forward noising process $q(x_t | x_(t-1))$ and reverse denoising process $p_theta (x_(t-1) | x_t)$ (Task 2).],
)[
  #fig-placeholder(height: 3.2cm)[
    Draw bidirectional diagram (Task 2 & Section IV): \
    Forward direction $x_0 -> x_1 -> dots.c -> x_T$ (adding Gaussian noise) and Reverse direction $x_T -> x_0$ (denoising via $epsilon_theta$).
  ]
] <fig:ddpm-chain>

== Step-by-Step Worked Numerical Example

#guideline("Task 3 & Section III — MANDATORY WORKED EXAMPLE")[
  - *Part A — 1D Single-Step Forward Noising (Task 3):*
    1. Pick concrete scalar values for $x_0$, $beta_1$, and $epsilon$.
    2. Manually compute conditional mean $sqrt(1 - beta_1) x_0$, standard deviation $sqrt(beta_1)$, and noisy output $x_1$.
    3. Interpret the stochastic expectation $bb(E)[x_1 | x_0]$ and variance preservation.
  - *Part B — Step-by-Step Loss Calculation for $L_("simple")$ (Section III):*
    1. Assume a concrete predicted value $epsilon_theta (x_1, 1)$ and compute the squared error $norm(epsilon - epsilon_theta (x_1, 1))^2$.
]

== Latent Diffusion Models (LDM)

#guideline("Task 5")[
  - *VAE Encoder / Decoder:* Explain spatial compression $z = cal(E)(x) in bb(R)^(h times w times c)$ with downsampling factor $f = H / h = W / w$ and reconstruction $tilde(x) = cal(D)(z_0)$.
  - *Operations in Latent Space:* Formulate forward/reverse diffusion on $z_t$ and explain why computational cost drops substantially.
  - *Cross-Attention Conditioning:* Explain domain encoder $tau_theta (y)$ and $"Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d)) V$ for text/layout prompts.
]

#figure(
  caption: [Overall pipeline of Latent Diffusion Models with VAE compression and cross-attention conditioning (Task 5).],
)[
  #fig-placeholder(height: 3.6cm)[
    Draw LDM architecture pipeline (Task 5 & Section IV): \
    Pixel Image $x ->$ VAE Encoder $cal(E) ->$ Latent Diffusion ($z_0 <-> z_T$, U-Net + Cross-Attention $tau_theta (y)$) $->$ VAE Decoder $cal(D) -> tilde(x)$.
  ]
] <fig:ldm-pipeline>

// ============================================================================
// 4. RESULTS AND DISCUSSION
// ============================================================================
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
  #fig-placeholder(height: 3.4cm)[
    Insert experimental image sequence (Task 4 & Section IV): \
    Row 1: Clean small image ($28 times 28$) + 5 increasing noise levels | Row 2: Pretrained denoising / generation example.
  ]
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

// ============================================================================
// 5. CONCLUSION
// ============================================================================
= Conclusion

#guideline("Section 5 | Target: ~0.5 page")[
  - Provide a concise answer to the *Guiding Question* from Section 1.
  - Summarize the progression from DDPM to LDM and mention 2–3 future directions (e.g., fast samplers, Diffusion Transformers).
]

// ============================================================================
// REFERENCES (EXCLUDED FROM THE 6–10 PAGE LIMIT)
// ============================================================================
#pagebreak()
#bibliography("references.bib", title: "References", style: "ieee")

// ============================================================================
// APPENDIX (OPTIONAL — EXCLUDED FROM THE 6–10 PAGE LIMIT)
// ============================================================================
#heading(numbering: none)[Appendix A. AI Tool Disclosure and Reproducibility]

#guideline("Appendix | Citation & Verification Requirement (Page 3 of Prompt)")[
  - *AI Tools Disclosure:* List any AI tools used during preparation and confirm independent verification of all claims, equations, and citations.
  - *Reproducibility:* Link to experiment notebooks/scripts and pretrained model checkpoints used.
]
