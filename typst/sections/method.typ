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

=== 1. VAE Encoder / Decoder & Perceptual Compression

To overcome the severe computational bottlenecks of pixel-space diffusion models, Latent Diffusion Models (LDM) introduce an explicit two-stage architecture that decouples perceptual compression from generative semantic modeling.

1. *Perceptual Spatial Compression ($z = cal(E)(x)$):* \
   An autoencoder consisting of an encoder $cal(E)$ compresses an RGB pixel image $x in bb(R)^(H times W times 3)$ into a lower-dimensional latent representation $z in bb(R)^(h times w times c)$:
   $ z = cal(E)(x) $
   The spatial dimensions are downsampled by a factor $f = H / h = W / w = 2^m$ (where $m in bb(N)$). LDM explores downsampling factors $f in \{1, 2, 4, 8, 16, 32\}$. For instance, an $f = 8$ compression factor reduces a $256 times 256 times 3$ pixel image to a $32 times 32 times 4$ latent tensor, discarding high-frequency imperceptible details while preserving essential structural semantics.

2. *Perceptual Reconstruction ($tilde(x) = cal(D)(z_0)$):* \
   A decoder $cal(D)$ learns to map the denoised latent representation $z_0$ back to the original image space:
   $ tilde(x) = cal(D)(z_0) = cal(D)(cal(E)(x)) $
   To prevent high-variance latent spaces during autoencoder training, two regularization variants are utilized:
   - *KL-reg:* Imposes a mild KL-penalty towards a standard normal distribution on the learned latent space, similar to a Variational Autoencoder.
   - *VQ-reg:* Incorporates a vector quantization layer within the decoder, acting as a VQ-GAN latent space. \
   Combined with perceptual loss terms (LPIPS) and patch-based adversarial losses, the autoencoder ensures that $tilde(x)$ maintains high visual fidelity without blurriness.

=== 2. Diffusion Operations in Latent Space & Computational Efficiency

By shifting the generative diffusion process into the lower-dimensional latent space $bb(R)^(h times w times c)$, LDM executes all forward noising and reverse denoising transitions on latent vectors $z_t$.

1. *Forward Latent Diffusion:* \
   $ q(z_(1:T) | z_0) = product_(t=1)^T q(z_t | z_(t-1)), quad q(z_t | z_0) = cal(N)(z_t; sqrt(overline(alpha)_t) z_0, (1 - overline(alpha)_t) bold(I)) $
   Arbitrary latent state sampling follows the closed form $z_t (z_0, epsilon) = sqrt(overline(alpha)_t) z_0 + sqrt(1 - overline(alpha)_t) epsilon$, where $epsilon ~ cal(N)(bold(0), bold(I))$.

2. *Reverse Latent Diffusion & Objective Function:* \
   The U-Net neural network $epsilon_theta (z_t, t, tau_theta (y))$ predicts the noise vector $epsilon$ injected into the latent space:
   $ L_"LDM" = bb(E)_(cal(E)(x), y, epsilon, t) [ norm(epsilon - epsilon_theta (z_t, t, tau_theta (y)))^2 ] $ <eq:lldm>

3. *Substantial Computational Savings:*
   - *Spatial Token Reduction:* Compressing image resolution by factor $f$ reduces spatial dimension size by $f^2$. For $f = 8$, the spatial token sequence length drops by a factor of $64$ (e.g., from $256^2 = 65,536$ pixels to $32^2 = 1,024$ latent tokens).
   - *Resource Efficiency:* Operating on $z_t$ lowers memory footprint and computational FLOPs for U-Net self-attention blocks, accelerating training convergence and boosting sampling throughput by $3 times$ to $10 times$ compared to pixel-space DDPM.

=== 3. Cross-Attention Conditioning Mechanism

To transform diffusion models into versatile, multi-modal conditional generators, LDM integrates a flexible Cross-Attention mechanism into the U-Net backbone.

1. *Domain-Specific Encoder ($tau_theta (y)$):* \
   Conditioning inputs $y$ (such as text prompts, layout bounding boxes, semantic segmentation maps, or target images) are processed by a specialized domain encoder $tau_theta (y) in bb(R)^(M times d_tau)$ that maps $y$ into an intermediate representation vector of sequence length $M$ and dimension $d_tau$ (e.g., a Transformer/BERT encoder for text).

2. *Cross-Attention Formulation:* \
   Let $phi_i (z_t) in bb(R)^(N times d_epsilon^i)$ denote the flattened intermediate spatial feature map of the U-Net at layer $i$, where $N = h_i times w_i$. The cross-attention layer projects features into Query, Key, and Value matrices:
   $ Q = W_Q^((i)) dot phi_i (z_t) in bb(R)^(N times d), quad K = W_K^((i)) dot tau_theta (y) in bb(R)^(M times d), quad V = W_V^((i)) dot tau_theta (y) in bb(R)^(M times d) $
   The attention weights and output feature updates are computed as:
   $ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d)) V $ <eq:crossattn>
   where $W_Q^((i)) in bb(R)^(d times d_epsilon^i)$, $W_K^((i)) in bb(R)^(d times d_tau)$, and $W_V^((i)) in bb(R)^(d times d_tau)$ are learnable projection weight matrices, and $d$ is the projection channel dimension.

3. *Multi-Modal Interaction:* \
   The matrix product $Q K^T in bb(R)^(N times M)$ evaluates pairwise affinity between each spatial location in the U-Net ($N$) and each conditioning token ($M$), enabling precise spatial alignment and semantic control (e.g., placing objects described in text at specific spatial regions).

=== 4. Overall Pipeline Diagram of Latent Diffusion Models

#figure(
  caption: [Overall pipeline of Latent Diffusion Models with VAE compression and cross-attention conditioning (Task 5).],
)[
  #rect(width: 100%, stroke: 0.6pt + luma(120), inset: 10pt, radius: 4pt, fill: luma(250))[
    #set text(size: 8.2pt)
    #grid(
      columns: (1fr),
      gutter: 8pt,
      [
        #align(center)[
          *1. Perceptual Compression & Reconstruction (Pixel <-> Latent)*
        ]
        #v(2pt)
        #grid(
          columns: (1.1fr, auto, 1.2fr, auto, 1.1fr),
          align: center + horizon,
          gutter: 4pt,
          [
            #rect(fill: rgb("e8f4f8"), stroke: 0.8pt + rgb("2b7bba"), radius: 3pt, inset: 5pt)[
              *Pixel Image $x$* \
              $H times W times 3$
            ]
          ],
          [
            #text(size: 10pt, fill: rgb("2b7bba"))[$-->$] \
            #text(size: 6.5pt, fill: rgb("2b7bba"))[Encoder $cal(E)$]
          ],
          [
            #rect(fill: rgb("fff8e7"), stroke: 0.8pt + rgb("d35400"), radius: 3pt, inset: 5pt)[
              *Latent Code $z_0$* \
              $h times w times c$ \
              #text(size: 6.5pt, fill: luma(100))[(Factor $f = H/h$)]
            ]
          ],
          [
            #text(size: 10pt, fill: rgb("27ae60"))[$-->$] \
            #text(size: 6.5pt, fill: rgb("27ae60"))[Decoder $cal(D)$]
          ],
          [
            #rect(fill: rgb("e8f8f5"), stroke: 0.8pt + rgb("27ae60"), radius: 3pt, inset: 5pt)[
              *Reconstructed $tilde(x)$* \
              $H times W times 3$
            ]
          ]
        )
      ],
      [
        #line(length: 100%, stroke: 0.4pt + luma(180))
      ],
      [
        #align(center)[
          *2. Latent Diffusion Process & Cross-Attention Conditioning*
        ]
        #v(2pt)
        #grid(
          columns: (1fr, auto, 1.3fr, auto, 1fr),
          align: center + horizon,
          gutter: 4pt,
          [
            #rect(fill: rgb("fff8e7"), stroke: 0.8pt + rgb("d35400"), radius: 3pt, inset: 5pt)[
              *Latent $z_0$* \
              $cal(E)(x)$
            ]
          ],
          [
            #text(size: 9pt, fill: rgb("c0392b"))[$<-->$] \
            #text(size: 6.5pt, fill: rgb("c0392b"))[Diffusion $q / p_theta$]
          ],
          [
            #rect(fill: rgb("f3e5f5"), stroke: 0.8pt + rgb("8e44ad"), radius: 3pt, inset: 6pt)[
              *Denoising U-Net $epsilon_theta (z_t, t, tau_theta (y))$* \
              #v(2pt)
              #rect(fill: rgb("ffffff"), stroke: 0.5pt + rgb("8e44ad"), radius: 2pt, inset: 3pt)[
                *Cross-Attention Layer* \
                $Q = W_Q phi_i (z_t), quad K = W_K tau_theta (y), quad V = W_V tau_theta (y)$ \
                $"Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d)) V$
              ]
            ]
          ],
          [
            #text(size: 9pt, fill: rgb("8e44ad"))[$<--$] \
            #text(size: 6.5pt, fill: rgb("8e44ad"))[$K, V$ Tokens]
          ],
          [
            #rect(fill: rgb("fef9e7"), stroke: 0.8pt + rgb("f39c12"), radius: 3pt, inset: 5pt)[
              *Conditioning $y$* \
              Text / Layout / Maps \
              #v(2pt)
              #text(size: 6.5pt)[Encoder $tau_theta (y)$]
            ]
          ]
        )
      ]
    )
    #v(4pt)
    #align(center)[
      #text(size: 7.5pt, style: "italic", fill: luma(80))[
        Full LDM Pipeline: Image $x$ is compressed into latent space $z_0 = cal(E)(x)$. Diffusion runs in latent space on $z_t$. Conditioning $y$ is encoded by $tau_theta (y)$ and injected via Cross-Attention layers inside the UNet $epsilon_theta$. Denoised $z_0$ is decoded back to pixel space $tilde(x) = cal(D)(z_0)$.
      ]
    ]
  ]
] <fig:ldm-pipeline>