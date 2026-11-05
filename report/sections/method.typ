#import "../helpers.typ": guideline, fig-placeholder

= Method and Worked Example

#guideline("Section 3 | Target: ~2.5–3.0 pages | Tasks 2, 3, 5 & Section III | Rubric: Algorithms & Equations (15 pts)")[
  Present the mathematical framework, symbol definitions, modeling assumptions, bidirectional diagrams, step-by-step numerical calculation, and LDM pipeline.
]

== DDPM: Forward Noising and Reverse Denoising

#guideline("Task 2 & Section III (Equations to Explain)")[
  - *Forward Process & $beta$ Schedule:* Explain progressive Gaussian noise injection via $beta_1, dots, beta_T$ (@eq:forward) and closed-form sampling $q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) bold(I))$.
  - *Reverse Process, Noise Predictor & Sampling:* Explain $p_theta (x_(t-1) | x_t)$, the U-Net noise predictor $epsilon_theta (x_t, t)$, the training loss (@eq:lsimple), and iterative sampling from $x_T ∼ cal(N)(bold(0), bold(I))$ back to $x_0$.
  - *Mandatory Specification (Section III):* Define *every symbol*, specify units/value ranges, and state modeling assumptions (@tab:symbols).
]

$ q(x_t | x_(t-1)) = cal(N)(sqrt(1 - beta_t) x_(t-1), beta_t bold(I)) $ <eq:forward>

$ L_"simple" = bb(E)_(t, x_0, epsilon) [ norm(epsilon - epsilon_theta (x_t, t))^2 ] $ <eq:lsimple>

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
    [$epsilon$], [_\[Define standard Gaussian noise\]_], [_\[$epsilon ∼ cal(N)(bold(0), bold(I))$\]_],
    [$epsilon_theta (x_t, t)$], [_\[Define neural noise predictor\]_], [_\[$bb(R)^D$\]_],
    table.hline(stroke: 0.9pt),
  )
] <tab:symbols>

#figure(
  caption: [Illustration of the forward noising process $q(x_t | x_(t-1))$ and reverse denoising process $p_theta (x_(t-1) | x_t)$ (Task 2).],
)[
  #fig-placeholder([
    Draw bidirectional diagram (Task 2 & Section IV): \
    Forward direction $x_0 -> x_1 -> dots.c -> x_T$ (adding Gaussian noise) and Reverse direction $x_T -> x_0$ (denoising via $epsilon_theta$).
  ], height: 3.2cm)
] <fig:ddpm-chain>

== Step-by-Step Worked Numerical Example

#guideline("Task 3 & Section III — MANDATORY WORKED EXAMPLE")[
  - *Part A — 1D Single-Step Forward Noising (Task 3):*
    1. Pick concrete scalar values for $x_0$, $beta_1$, and $epsilon$.
    2. Manually compute conditional mean $sqrt(1 - beta_1) x_0$, standard deviation $sqrt(beta_1)$, and noisy output $x_1$.
    3. Interpret the stochastic expectation $bb(E)[x_1 | x_0]$ and variance preservation.
  - *Part B — Step-by-Step Loss Calculation for $L_"simple"$ (Section III):*
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
  #fig-placeholder([
    Draw LDM architecture pipeline (Task 5 & Section IV): \
    Pixel Image $x ->$ VAE Encoder $cal(E) ->$ Latent Diffusion ($z_0 <-> z_T$, U-Net + Cross-Attention $tau_theta (y)$) $->$ VAE Decoder $cal(D) -> tilde(x)$.
  ], height: 3.6cm)
] <fig:ldm-pipeline>
