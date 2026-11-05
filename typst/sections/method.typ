#import "../helpers.typ": guideline, fig-placeholder

= Method and Worked Example

#guideline("Section 3 | Target: ~2.5–3.0 pages | Tasks 2, 3, 5 & Section III | Rubric: Algorithms & Equations (15 pts)")[
  Present the mathematical framework, symbol definitions, modeling assumptions, bidirectional diagrams, step-by-step numerical calculation, and LDM pipeline.
]

== DDPM: Forward Noising and Reverse Denoising

#guideline("Task 2 & Section III (Equations to Explain)")[
  - *Forward Process & $beta$ Schedule:* Explain progressive Gaussian noise injection via $beta_1, dots, beta_T$ (@eq:forward) and closed-form sampling $q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) bold(I))$.
  - *Reverse Process, Noise Predictor & Sampling:* Explain $p_theta (x_(t-1) | x_t)$, the U-Net noise predictor $epsilon_theta (x_t, t)$, the training loss (@eq:lsimple), and iterative sampling from $x_T ~ cal(N)(bold(0), bold(I))$ back to $x_0$.
  - *Mandatory Specification (Section III):* Define *every symbol*, specify units/value ranges, and state modeling assumptions (@tab:symbols).
]

=== 1. Theoretical Framework & Modeling Assumptions

Denoising Diffusion Probabilistic Models (DDPM) define a class of latent variable models inspired by nonequilibrium thermodynamics. The generative mechanism consists of two opposing Markov chains operating over a high-dimensional continuous space $bb(R)^D$: a parameter-free *forward diffusion process* that systematically degrades data into isotropic Gaussian noise, and a parameterized *reverse process* that learns to invert this degradation to synthesize novel samples.

==== Data Scaling & Preprocessing Assumptions

Image data consisting of discrete pixel intensities $x_"raw" in {0, 1, ..., 255}^D$ (where $D = H times W times C$) is scaled linearly to the continuous interval $[-1, 1]^D$:

$ x_0 = (x_"raw") / 127.5 - 1 $

This normalization ensures that the input signal $x_0$ is centered at zero with coordinate-wise variance consistent with the standard Gaussian prior $p(x_T) = cal(N)(bold(0), bold(I))$, preventing numerical instabilities during neural network processing.

==== Forward Process & $beta$ Schedule

The forward diffusion process $q(x_(1:T) | x_0)$ is defined as a fixed Markov chain that progressively corrupts data by adding small amounts of Gaussian noise at each discrete timestep $t in {1, ..., T}$ according to a variance schedule $beta_1, ..., beta_T$:

$ q(x_(1:T) | x_0) = product_(t=1)^T q(x_t | x_(t-1)), quad q(x_t | x_(t-1)) = cal(N)(x_t; sqrt(1 - beta_t) x_(t-1), beta_t bold(I)) $ <eq:forward>

The variance hyperparameters are constrained to $beta_t in (0, 1)$ and increase linearly across $T = 1000$ timesteps, ranging from $beta_1 = 10^(-4)$ to $beta_T = 0.02$. Choosing small $beta_t$ relative to the unit-scaled data bounds the entropy change per step, ensuring that the reverse transitions retain the same functional Gaussian form as the forward steps.

A key property of the forward process is its ability to sample the noisy state $x_t$ at any arbitrary timestep $t$ directly from $x_0$ in closed form. Defining $alpha_t := 1 - beta_t$ and $overline(alpha)_t := product_(s=1)^t alpha_s$, the conditional distribution marginalizes to:

$ q(x_t | x_0) = cal(N)(x_t; sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) bold(I)) $ <eq:closedform>

Using the reparameterization trick, any noisy state $x_t$ can be expressed directly as a linear combination of the clean signal $x_0$ and a standard Gaussian noise vector $epsilon ~ cal(N)(bold(0), bold(I))$:

$ x_t (x_0, epsilon) = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon $ <eq:reparam>

As $t -> T$, $overline(alpha)_T -> 0$, causing the signal $x_0$ to vanish completely so that $x_T approx cal(N)(bold(0), bold(I))$.

==== Reverse Process, Noise Predictor & Sampling

The reverse process $p_theta (x_(0:T))$ is a learned Markov chain starting from a standard Gaussian prior $p(x_T) = cal(N)(x_T; bold(0), bold(I))$. It is parameterized by a U-Net neural network with parameters $theta$:

$ p_theta (x_(0:T)) = p(x_T) product_(t=1)^T p_theta (x_(t-1) | x_t), quad p_theta (x_(t-1) | x_t) = cal(N)(x_(t-1); mu_theta (x_t, t), Sigma_theta (x_t, t)) $ <eq:reverse>

To parameterize $p_theta (x_(t-1) | x_t)$, the covariance matrix is set to untrainable time-dependent constants $Sigma_theta (x_t, t) = sigma_t^2 bold(I)$, where $sigma_t^2$ is fixed to either $beta_t$ (upper bound on entropy) or $tilde(beta)_t = ((1 - overline(alpha)_(t-1)) / (1 - overline(alpha)_t)) beta_t$ (lower bound for deterministic $x_0$).

When conditioned on $x_0$, the tractable posterior $q(x_(t-1) | x_t, x_0) = cal(N)(x_(t-1); tilde(mu)_t (x_t, x_0), tilde(beta)_t bold(I))$ has a mean of:

$ tilde(mu)_t (x_t, x_0) = (sqrt(overline(alpha)_(t-1)) beta_t) / (1 - overline(alpha)_t) x_0 + (sqrt(alpha_t)(1 - overline(alpha)_(t-1))) / (1 - overline(alpha)_t) x_t $

Substituting $x_0 = (x_t - sqrt(1 - overline(alpha)_t) epsilon) / (sqrt(overline(alpha)_t))$ into $tilde(mu)_t$ reveals that the neural network $mu_theta (x_t, t)$ should be reparameterized to predict the noise $epsilon$ rather than the mean directly:

$ mu_theta (x_t, t) = 1 / (sqrt(alpha_t)) (x_t - (beta_t) / (sqrt(1 - overline(alpha)_t)) epsilon_theta (x_t, t)) $ <eq:mu_param>

This reparameterization simplifies the variational lower bound (ELBO) into an unweighted mean squared error objective over noise vectors, termed $L_"simple"$:

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
    [$x_0$], [[Define clean normalized data]], [[e.g., $[-1, 1]^D$]],
    [$x_t, x_(t-1)$], [[Define noisy states at $t, t-1$]], [[e.g., $bb(R)^D$]],
    [$t, T$], [[Define discrete timestep and total steps]], [[e.g., $t in {1, ..., T}$]],
    [$beta_t$], [[Define variance schedule at step $t$]], [[e.g., $beta_t in (0, 1)$]],
    [$bold(I)$], [[Define identity covariance matrix]], [[$bb(R)^(D times D)$]],
    [$epsilon$], [[Define standard Gaussian noise]], [[$epsilon ~ cal(N)(bold(0), bold(I))$]],
    [$epsilon_theta (x_t, t)$], [[Define neural noise predictor]], [[$bb(R)^D$]],
    table.hline(stroke: 0.9pt),
  )
] <tab:symbols>

#figure(
  image("../figures/fig1_ddpm_chain.pdf", width: 100%),
  caption: [Illustration of the forward noising process $q(x_t | x_(t-1))$ and reverse denoising process $p_theta (x_(t-1) | x_t)$ (Task 2).],
) <fig:ddpm-chain>

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