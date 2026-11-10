#import "../helpers.typ": guideline, fig-placeholder

#set math.equation(numbering: "(1)")

= Method and Worked Example



== DDPM: Forward Noising and Reverse Denoising


=== Theoretical Framework & Modeling Assumptions

Denoising Diffusion Probabilistic Models (DDPM) define a class of latent variable models inspired by nonequilibrium thermodynamics. The generative mechanism consists of two opposing Markov chains operating over a high-dimensional continuous space $bb(R)^D$: a parameter-free *forward diffusion process* that systematically degrades data into isotropic Gaussian noise, and a parameterized *reverse process* that learns to invert this degradation to synthesize novel samples.

=== Data Scaling & Preprocessing Assumptions

Image data consisting of discrete pixel intensities $x_"raw" in {0, 1, ..., 255}^D$ (where $D = H times W times C$) is scaled linearly to the continuous interval $[-1, 1]^D$:

$ x_0 = (x_"raw") / 127.5 - 1 $

This normalization ensures that the input signal $x_0$ is centered at zero with coordinate-wise variance consistent with the standard Gaussian prior $p(x_T) = cal(N)(bold(0), bold(I))$, preventing numerical instabilities during neural network processing.

=== Forward Process & $beta$ Schedule

The forward diffusion process $q(x_(1:T) | x_0)$ is defined as a fixed Markov chain that progressively corrupts data by adding small amounts of Gaussian noise at each discrete timestep $t in {1, ..., T}$ according to a variance schedule $beta_1, ..., beta_T$:

$ q(x_(1:T) | x_0) = product_(t=1)^T q(x_t | x_(t-1)), quad q(x_t | x_(t-1)) = cal(N)(x_t; sqrt(1 - beta_t) x_(t-1), beta_t bold(I)) $ <eq:forward>

The variance hyperparameters are constrained to $beta_t in (0, 1)$ and increase linearly across $T = 1000$ timesteps, ranging from $beta_1 = 10^(-4)$ to $beta_T = 0.02$. Choosing small $beta_t$ relative to the unit-scaled data bounds the entropy change per step, ensuring that the reverse transitions retain the same functional Gaussian form as the forward steps.

A key property of the forward process is its ability to sample the noisy state $x_t$ at any arbitrary timestep $t$ directly from $x_0$ in closed form. Defining $alpha_t := 1 - beta_t$ and $overline(alpha)_t := product_(s=1)^t alpha_s$, the conditional distribution marginalizes to:

$ q(x_t | x_0) = cal(N)(x_t; sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) bold(I)) $ <eq:closedform>

Using the reparameterization trick, any noisy state $x_t$ can be expressed directly as a linear combination of the clean signal $x_0$ and a standard Gaussian noise vector $epsilon ~ cal(N)(bold(0), bold(I))$:

$ x_t (x_0, epsilon) = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon $ <eq:reparam>

As $t -> T$, $overline(alpha)_T -> 0$, causing the signal $x_0$ to vanish completely so that $x_T approx cal(N)(bold(0), bold(I))$.

=== Reverse Process, Noise Predictor & Sampling

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
  image("../../figures/fig1_ddpm_chain.pdf", width: 100%),
  caption: [Illustration of the forward noising process $q(x_t | x_(t-1))$ and reverse denoising process $p_theta (x_(t-1) | x_t)$ (Task 2).],
) <fig:ddpm-chain>

== Step-by-Step Worked Numerical Example



The numerical example uses the MNIST test set and selects the first image labeled as the digit `3`, whose raw pixel intensities $I_0 in [0, 255]$ are normalized to $x_0 in [-1, 1]$ using $x_0 = I_0 / 127.5 - 1$; the variance schedule follows the method of Ho and colleagues (2020) with $T = 1000$, $beta_1 = 0.0001$, and $beta_T = 0.02$, and a fixed random seed of `42` is used for both NumPy and PyTorch, while the calculation uses the pixel at row $10$ and column $14$ from the central foreground region; its raw intensity is $236$ and its normalized value is:

$ x_0 = frac(236, 127.5) - 1 = 0.850980 $ <eq-x0>

The fixed seed produces Gaussian noise $epsilon = -0.714962$ at this pixel, and at timestep $t = 1$ the signal coefficient is $sqrt(1 - beta_1) = sqrt(0.9999) approx 0.999950$ while the noise standard deviation is $sqrt(beta_1) = 0.010000$; consequently, the conditional mean is $mu_1 = 0.999950 times 0.850980 = 0.850938$ and the added noise is $0.010000 times (-0.714962) = -0.007150$, yielding the noisy pixel value:

$ x_1 = mu_1 + sqrt(beta_1) epsilon = 0.850938 - 0.007150 = 0.843788 $ <eq-x1>

This result is one realization from $q(x_1 | x_0) = cal(N)(sqrt(1 - beta_1) x_0, beta_1 I)$, whose expectation is $bb(E)[x_1 | x_0] = 0.850938$; for a data distribution with unit variance and independent unit-variance noise, the marginal variance remains $(1 - beta_1) + beta_1 = 1$, while the signal contribution gradually decreases over the diffusion steps; the closed-form expression gives a later state without simulating every intermediate transition, and at $t = 50$, where $bar(alpha)_(50) = 0.971016$, the signal and noise coefficients are $0.985401$ and $0.170248$, respectively, so the noisy state is:

$ x_(50) = 0.985401 times 0.850980 + 0.170248 times (-0.714962) = 0.716836 $

Using the same example, the simplified noise-prediction objective at timestep $t = 1$ is:

$ L_"simple" = EE_(t, x_0, epsilon) [ norm(epsilon - epsilon_theta(x_t, t))^2 ] $ <eq-lsimple>

Passing $x_1$ through the initial, untrained `ConditionalUNet` with time and class conditioning gives predicted noise $epsilon_theta(x_1, 1) = 0.421943$, so the difference between the injected noise and the prediction is:

$ e_1 = epsilon - epsilon_theta(x_1, 1) = -0.714962 - 0.421943 = -1.136905 $

Thus, the pointwise squared error for the selected pixel is:

$ norm(epsilon - epsilon_theta(x_1, 1))^2 = (-1.136905)^2 = 1.292554 $

For comparison, averaging the squared prediction errors across all $28 times 28 = 784$ pixels gives the image-level loss:

$ L_"simple"^(t=1) = frac(1, 784) sum_(r=1)^(28) sum_(c=1)^(28) (epsilon_(r, c) - epsilon_theta(x_1, 1)_(r, c))^2 = 1.134913 $

The single-pixel error is one contribution to this image-level mean, while training minimizes the corresponding noise-prediction error over sampled images, timesteps, and noise vectors

== Latent Diffusion Models (LDM) <sec:ldm>



To overcome the severe computational bottlenecks of pixel-space diffusion models, Latent Diffusion Models (LDM) introduce an explicit two-stage architecture that decouples perceptual compression from generative semantic modeling

An autoencoder maps an RGB image $x in bb(R)^(H times W times 3)$ into a lower-dimensional latent representation $z in bb(R)^(h times w times c)$ with its encoder $cal(E)$

$ z = cal(E)(x) $ <eq:vae-enc>

The spatial downsampling factor is $f = H / h = W / w = 2^m$, where $m in bb(N)$ and LDM considers factors $f in \{1, 2, 4, 8, 16, 32\}$; for instance, with $f = 8$, an image of size $256 times 256 times 3$ becomes a latent tensor of size $32 times 32 times 4$, reducing computation while retaining important structural information, and a decoder maps the denoised latent $z_0$ back to image space

$ tilde(x) = cal(D)(z_0) = cal(D)(cal(E)(x)) $ <eq:vae-dec>

To keep the latent representation useful for reconstruction, the autoencoder can use KL regularization toward a standard normal distribution or vector quantization as in VQ-GAN, alongside perceptual losses such as LPIPS and patch-based adversarial losses

LDM applies the forward and reverse diffusion processes to latent states $z_t$ rather than full-resolution pixels, with the forward process defined as

$ q(z_(1:T) | z_0) = product_(t=1)^T q(z_t | z_(t-1)), quad q(z_t | z_0) = cal(N)(z_t; sqrt(overline(alpha)_t) z_0, (1 - overline(alpha)_t) bold(I)) $

The closed-form expression $z_t (z_0, epsilon) = sqrt(overline(alpha)_t) z_0 + sqrt(1 - overline(alpha)_t) epsilon$, where $epsilon ~ cal(N)(bold(0), bold(I))$, gives the latent state at any timestep, and during denoising, the U-Net $epsilon_theta (z_t, t, tau_theta (y))$ estimates the noise conditioned on the encoded input $y$, with the model trained using the objective

$ L_"LDM" = bb(E)_(cal(E)(x), y, epsilon, t) [ norm(epsilon - epsilon_theta (z_t, t, tau_theta (y)))^2 ] $ <eq:lldm>

Compressing each spatial dimension by $f$ reduces the number of spatial tokens by $f^2$; at $f = 8$, for instance, a $256 times 256$ image has $65,536$ spatial positions while its $32 times 32$ latent has $1,024$, lowering memory use and computation in the U-Net and its attention layers and allowing faster training and sampling than pixel-space diffusion

For conditional generation, LDM encodes text prompts, layouts, segmentation maps, or other inputs with a domain-specific encoder $tau_theta (y) in bb(R)^(M times d_tau)$, producing $M$ conditioning tokens of dimension $d_tau$; at U-Net layer $i$, the spatial features $phi_i (z_t) in bb(R)^(N times d_epsilon^i)$, where $N = h_i times w_i$, are projected into queries, while the encoded conditioning tokens provide keys and values:

$ Q = W_Q^((i)) dot phi_i (z_t) in bb(R)^(N times d), quad K = W_K^((i)) dot tau_theta (y) in bb(R)^(M times d), quad V = W_V^((i)) dot tau_theta (y) in bb(R)^(M times d) $

Cross-attention is calculated as

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d)) V $ <eq:crossattn>

Here, $W_Q^((i))$, $W_K^((i))$, and $W_V^((i))$ are learned projection matrices and $d$ is the projection dimension; the matrix $Q K^T in bb(R)^(N times M)$ measures the relationship between each spatial location and each conditioning token, allowing the generated content to reflect the input prompt or layout

#figure(
  caption: [Overall pipeline of Latent Diffusion Models with VAE compression and cross-attention conditioning (Task 5)],
)[
  #rect(width: 100%, stroke: 0.8pt + luma(50), inset: 8pt, radius: 4pt, fill: luma(252))[
    #set text(size: 8pt, fill: luma(20))
    #grid(
      columns: (1fr),
      gutter: 8pt,
      [
        #align(center)[
          #text(weight: "bold", size: 8.5pt)[1. Perceptual Compression (Pixel Space $arrow.r.l$ Latent Space)]
        ]
        #v(1pt)
        #grid(
          columns: (1.1fr, auto, 1.1fr, auto, 1.1fr),
          align: center + horizon,
          gutter: 6pt,
          [
            #rect(fill: luma(245), stroke: 0.7pt + luma(50), radius: 3pt, inset: 5pt)[
              *Image $x$* \
              #text(size: 7pt)[$H times W times 3$]
            ]
          ],
          [
            #text(size: 9pt)[$-->$] \
            #text(size: 7pt)[$cal(E)$ (Encoder)]
          ],
          [
            #rect(fill: luma(238), stroke: 0.9pt + luma(30), radius: 3pt, inset: 5pt)[
              *Latent $z_0$* \
              #text(size: 7pt)[$h times w times c$]
            ]
          ],
          [
            #text(size: 9pt)[$-->$] \
            #text(size: 7pt)[$cal(D)$ (Decoder)]
          ],
          [
            #rect(fill: luma(245), stroke: 0.7pt + luma(50), radius: 3pt, inset: 5pt)[
              *Output $tilde(x)$* \
              #text(size: 7pt)[$H times W times 3$]
            ]
          ]
        )
      ],
      [
        #line(length: 100%, stroke: 0.5pt + luma(180))
      ],
      [
        #align(center)[
          #text(weight: "bold", size: 8.5pt)[2. Latent Diffusion & Cross-Attention Conditioning]
        ]
        #v(1pt)
        #grid(
          columns: (0.9fr, auto, 1.4fr, auto, 0.9fr),
          align: center + horizon,
          gutter: 6pt,
          [
            #rect(fill: luma(238), stroke: 0.9pt + luma(30), radius: 3pt, inset: 5pt)[
              *Latent $z_0$* \
              #text(size: 7pt)[$z_0 = cal(E)(x)$]
            ]
          ],
          [
            #text(size: 9pt)[$<-->$] \
            #text(size: 7pt)[Diffusion $q / p_theta$]
          ],
          [
            #rect(fill: luma(245), stroke: 0.7pt + luma(50), radius: 3pt, inset: 6pt)[
              *Denoising U-Net $epsilon_theta(z_t, t)$* \
              #v(2pt)
              #rect(fill: luma(255), stroke: 0.5pt + luma(100), radius: 2pt, inset: 3pt)[
                *Cross-Attention* \
                #text(size: 7pt)[$Q = W_Q phi(z_t), quad K, V = W_(K,V) tau_theta(y)$]
              ]
            ]
          ],
          [
            #text(size: 9pt)[$<--$] \
            #text(size: 7pt)[$K, V$]
          ],
          [
            #rect(fill: luma(245), stroke: 0.7pt + luma(50), radius: 3pt, inset: 5pt)[
              *Condition $y$* \
              #text(size: 7pt)[Encoder $tau_theta(y)$]
            ]
          ]
        )
      ]
    )
  ]
] <fig:ldm-pipeline>