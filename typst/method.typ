#import "../helpers.typ": guideline, fig-placeholder

= Method and Worked Example

== DDPM: Forward Noising and Reverse Denoising

=== 1. Theoretical Framework & Modeling Assumptions

Denoising Diffusion Probabilistic Models (DDPM) define a class of latent variable models inspired by nonequilibrium thermodynamics. The generative mechanism consists of two opposing Markov chains operating over a high-dimensional continuous space $\bb(R)^D$: a parameter-free *forward diffusion process* that systematically degrades data into isotropic Gaussian noise, and a parameterized *reverse process* that learns to invert this degradation to synthesize novel samples.

==== Data Scaling & Preprocessing Assumptions
Image data consisting of discrete pixel intensities $x_{\text{raw}} \in \{0, 1, \dots, 255\}^D$ (where $D = H \times W \times C$) is scaled linearly to the continuous interval $[-1, 1]^D$:
$ x_0 = \frac{x_{\text{raw}}}{127.5} - 1 $
This normalization ensures that the input signal $x_0$ is centered at zero with coordinate-wise variance consistent with the standard Gaussian prior $p(x_T) = cal(N)(bold(0), bold(I))$, preventing numerical instabilities during neural network processing.

==== Forward Process & $beta$ Schedule
The forward diffusion process $q(x_{1:T} | x_0)$ is defined as a fixed Markov chain that progressively corrupts data by adding small amounts of Gaussian noise at each discrete timestep $t \in \{1, \dots, T\}$ according to a variance schedule $\beta_1, \dots, \beta_T$:
$ q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1}), \quad q(x_t | x_(t-1)) = cal(N)(x_t; sqrt(1 - beta_t) x_(t-1), beta_t bold(I)) $ <eq:forward>

The variance hyperparameters are constrained to $\beta_t \in (0, 1)$ and increase linearly across $T = 1000$ timesteps, ranging from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. Choosing small $\beta_t$ relative to the unit-scaled data bounds the entropy change per step, ensuring that the reverse transitions retain the same functional Gaussian form as the forward steps.

A key property of the forward process is its ability to sample the noisy state $x_t$ at any arbitrary timestep $t$ directly from $x_0$ in closed form. Defining $\alpha_t := 1 - \beta_t$ and $\overline{\alpha}_t := \prod_{s=1}^t \alpha_s$, the conditional distribution marginalizes to:
$ q(x_t | x_0) = cal(N)(x_t; sqrt(overline(\alpha)_t) x_0, (1 - overline(\alpha)_t) bold(I)) $ <eq:closedform>
Using the reparameterization trick, any noisy state $x_t$ can be expressed directly as a linear combination of the clean signal $x_0$ and a standard Gaussian noise vector $\epsilon \sim cal(N)(bold(0), bold(I))$:
$ x_t(x_0, \epsilon) = sqrt(overline(\alpha)_t) x_0 + sqrt(1 - overline(\alpha)_t) \epsilon $ <eq:reparam>

As $t \to T$, $\overline{\alpha}_T \to 0$, causing the signal $x_0$ to vanish completely so that $x_T \approx cal(N)(bold(0), bold(I))$.

==== Reverse Process, Noise Predictor & Sampling
The reverse process $p_theta(x_{0:T})$ is a learned Markov chain starting from a standard Gaussian prior $p(x_T) = cal(N)(x_T; bold(0), bold(I))$. It is parameterized by a U-Net neural network with parameters $\theta$:
$ p_theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_theta(x_{t-1} | x_t), \quad p_theta(x_{t-1} | x_t) = cal(N)(x_{t-1}; \mu_theta(x_t, t), \Sigma_theta(x_t, t)) $ <eq:reverse>

To parameterize $p_theta(x_{t-1} | x_t)$, the covariance matrix is set to untrainable time-dependent constants $\Sigma_theta(x_t, t) = \sigma_t^2 bold(I)$, where $\sigma_t^2$ is fixed to either $\beta_t$ (upper bound on entropy) or $\tilde{\beta}_t = \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t} \beta_t$ (lower bound for deterministic $x_0$).

When conditioned on $x_0$, the tractable posterior $q(x_{t-1} | x_t, x_0) = cal(N)(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t bold(I))$ has a mean of:
$ \tilde{\mu}_t(x_t, x_0) = \frac{sqrt(overline(\alpha)_{t-1}) \beta_t}{1 - overline(\alpha)_t} x_0 + \frac{sqrt(\alpha_t) (1 - overline(\alpha)_{t-1})}{1 - overline(\alpha)_t} x_t $
Substituting $x_0 = \frac{x_t - sqrt(1 - overline(\alpha)_t) \epsilon}{sqrt(overline(\alpha)_t)}$ into $\tilde{\mu}_t$ reveals that the neural network $\mu_theta(x_t, t)$ should be reparameterized to predict the noise $\epsilon$ rather than the mean directly:
$ \mu_theta(x_t, t) = \frac{1}{sqrt(\alpha_t)} \left( x_t - \frac{\beta_t}{sqrt(1 - overline(\alpha)_t)} \epsilon_theta(x_t, t) \right) $ <eq:mu_param>

This reparameterization simplifies the variational lower bound (ELBO) into an unweighted mean squared error objective over noise vectors, termed $L_{\text{simple}}$:
$ L_{\text{simple}} = bb(E)_(t, x_0, \epsilon) \left[ \norm{\epsilon - \epsilon_theta(x_t, t)}^2 \right] $ <eq:lsimple>
where $t \sim \text{Uniform}(\{1, \dots, T\})$, $x_0 \sim q(x_0)$, and $\epsilon \sim cal(N)(bold(0), bold(I))$. Minimizing $L_{\text{simple}}$ downweights terms corresponding to small $t$, allowing the model to focus on harder denoising tasks at higher noise levels, which substantially improves sample fidelity.

Sampling proceeds iteratively from $x_T \sim cal(N)(bold(0), bold(I))$ back to $x_0$:
$ x_{t-1} = \frac{1}{sqrt(\alpha_t)} \left( x_t - \frac{\beta_t}{sqrt(1 - overline(\alpha)_t)} \epsilon_theta(x_t, t) \right) + \sigma_t z, \quad z \sim cal(N)(bold(0), bold(I)) \text{ if } t > 1 \text{ else } bold(0) $

=== 2. Mathematical Symbol Specifications & Modeling Assumptions

#figure(
  caption: [Symbol definitions, value ranges, and roles in @eq:forward and @eq:lsimple (Section III).],
)[
  #set text(size: 8.8pt)
  #table(
    columns: (1fr, 2.3fr, 1.3fr),
    align: (left, left, left),
    stroke: none,
    inset: (x: 5pt, y: 5pt),
    table.hline(stroke: 0.9pt),
    [*Symbol*], [*Definition & Role*], [*Range / Units*],
    table.hline(stroke: 0.5pt),
    [$x_0$], [Clean normalized image data; target distribution input signal.], [$[-1, 1]^D \quad (D = H \times W \times C)$],
    [$x_t, x_(t-1)$], [Noisy intermediate latent state vectors at timesteps $t$ and $t-1$.], [$\bb(R)^D$],
    [$t, T$], [Discrete timestep index ($t$) and total diffusion steps ($T = 1000$).], [$t \in \{1, \dots, T\}, \quad T \in \bb(N)^+$],
    [$beta_t$], [Forward variance schedule parameter at timestep $t$.], [$\beta_t \in (10^{-4}, 0.02) \subset (0, 1)$],
    [$alpha_t$], [Fraction of retained signal variance at step $t$ ($\alpha_t = 1 - \beta_t$).], [$\alpha_t \in (0.98, 0.9999) \subset (0, 1)$],
    [$overline(alpha)_t$], [Cumulative signal variance product ($\overline{\alpha}_t = \prod_{s=1}^t \alpha_s$).], [$\overline{\alpha}_t \in (0, 1), \quad \overline{\alpha}_T \approx 0$],
    [$bold(I)$], [Identity covariance matrix enforcing isotropic Gaussian noise.], [$\bb(R)^(D \times D)$],
    [$epsilon$], [Uncorrelated standard Gaussian noise added in forward step.], [$\epsilon \sim cal(N)(bold(0), bold(I)) \in \bb(R)^D$],
    [$epsilon_theta (x_t, t)$], [U-Net neural noise predictor estimating injected noise $\epsilon$.], [$\bb(R)^D$],
    [$mu_theta (x_t, t)$], [Predicted Gaussian distribution mean for reverse transition $p_theta$.], [$\bb(R)^D$],
    [$sigma_t^2$], [Fixed variance for reverse step $p_theta$ ($\sigma_t^2 = \beta_t$ or $\tilde{\beta}_t$).], [$\sigma_t^2 \in \bb(R)^+$],
    table.hline(stroke: 0.9pt),
  )
] <tab:symbols>

=== 3. Bidirectional Diffusion Diagram

#figure(
  caption: [Illustration of the forward noising process $q(x_t | x_(t-1))$ and reverse denoising process $p_theta (x_(t-1) | x_t)$ (Task 2).],
)[
  #rect(width: 100%, stroke: 0.6pt + luma(120), inset: 10pt, radius: 4pt, fill: luma(250))[
    #set text(size: 8.5pt)
    #grid(
      columns: (1fr, auto, 1.2fr, auto, 1.2fr, auto, 1.fr),
      align: center + horizon,
      gutter: 3pt,
      [
        #rect(fill: rgb("e8f4f8"), stroke: 0.8pt + rgb("2b7bba"), radius: 3pt, inset: 6pt)[
          *Clean Signal*          $x_0 \in [-1,1]^D$
        ]
      ],
      [
        #text(size: 11pt, fill: rgb("c0392b"))[$-->$]        #text(size: 7pt, fill: rgb("c0392b"))[$q(x_1|x_0)$]
      ],
      [
        #rect(fill: rgb("f9f9f9"), stroke: 0.6pt + luma(150), radius: 3pt, inset: 5pt)[
          *Slight Noise*          $x_1 \sim q(x_1|x_0)$
        ]
      ],
      [
        #text(size: 11pt, fill: rgb("c0392b"))[$--> \dots.c -->$]        #text(size: 7pt, fill: rgb("c0392b"))[$q(x_t|x_(t-1))$]
      ],
      [
        #rect(fill: rgb("fff8e7"), stroke: 0.8pt + rgb("d35400"), radius: 3pt, inset: 5pt)[
          *Intermediate*          $x_t = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\epsilon$
        ]
      ],
      [
        #text(size: 11pt, fill: rgb("c0392b"))[$--> \dots.c -->$]        #text(size: 7pt, fill: rgb("c0392b"))[$q(x_T|x_(T-1))$]
      ],
      [
        #rect(fill: rgb("f4f4f4"), stroke: 0.8pt + luma(80), radius: 3pt, inset: 6pt)[
          *Pure Noise*          $x_T \sim cal(N)(bold(0), bold(I))$
        ]
      ]
    )

    #v(6pt)
    #line(length: 100%, stroke: 0.4pt + luma(180))
    #v(6pt)

    #grid(
      columns: (1fr, auto, 1.2fr, auto, 1.2fr, auto, 1.fr),
      align: center + horizon,
      gutter: 3pt,
      [
        #rect(fill: rgb("e8f8f5"), stroke: 0.8pt + rgb("27ae60"), radius: 3pt, inset: 6pt)[
          *Sample*          $x_0$
        ]
      ],
      [
        #text(size: 11pt, fill: rgb("27ae60"))[$<--$]        #text(size: 7pt, fill: rgb("27ae60"))[$p_theta(x_0|x_1)$]
      ],
      [
        #rect(fill: rgb("f9f9f9"), stroke: 0.6pt + luma(150), radius: 3pt, inset: 5pt)[
          *Denoised*          $x_1 \sim p_theta(x_1|x_2)$
        ]
      ],
      [
        #text(size: 11pt, fill: rgb("27ae60"))[$<-- \dots.c <--$]        #text(size: 7pt, fill: rgb("27ae60"))[$p_theta(x_(t-1)|x_t)$]
      ],
      [
        #rect(fill: rgb("f3e5f5"), stroke: 0.8pt + rgb("8e44ad"), radius: 3pt, inset: 5pt)[
          *U-Net Predictor*          $\epsilon_theta(x_t, t)$
        ]
      ],
      [
        #text(size: 11pt, fill: rgb("27ae60"))[$<-- \dots.c <--$]        #text(size: 7pt, fill: rgb("27ae60"))[$p_theta(x_(T-1)|x_T)$]
      ],
      [
        #rect(fill: rgb("f4f4f4"), stroke: 0.8pt + luma(80), radius: 3pt, inset: 6pt)[
          *Gaussian Prior*          $x_T \sim cal(N)(bold(0), bold(I))$
        ]
      ]
    )
    #v(4pt)
    #align(center)[
      #text(size: 7.5pt, style: "italic", fill: luma(80))[
        Top row: Forward process $q(x_t | x_(t-1))$ destroys signal by adding noise $\epsilon$. Bottom row: Reverse process $p_theta(x_(t-1) | x_t)$ generates images by predicting noise via U-Net $\epsilon_theta(x_t, t)$.
      ]
    ]
  ]
] <fig:ddpm-chain>

== Step-by-Step Worked Numerical Example

=== Part A — 1D Single-Step Forward Noising (Task 3)

To illustrate the forward diffusion mechanics concretely, consider a one-dimensional continuous scalar space ($D = 1$). We select specific, realistic numerical values for the clean data point $x_0$, the step-1 variance parameter $\beta_1$, and the standard Gaussian noise sample $\epsilon$:

1. *Concrete Input Parameter Selection:*
   - Clean data signal: $x_0 = 2.0$ (representing a normalized pixel intensity scaled to $[-1, 1]$).
   - Step 1 variance schedule value: $\beta_1 = 0.04$ (hence $\alpha_1 = 1 - \beta_1 = 0.96$).
   - Standard Gaussian noise sample: $\epsilon = 0.5 \sim cal(N)(0, 1)$.

2. *Manual Step-by-Step Computations:*
   - *Conditional Mean ($\mu_{1|0}$):*
     $ \mu_{1|0} = sqrt(1 - beta_1) x_0 = sqrt(0.96) \times 2.0 \approx 0.9797959 \times 2.0 = 1.9595918 $
   - *Conditional Standard Deviation ($\sigma_{1|0}$):*
     $ \sigma_{1|0} = sqrt(beta_1) = sqrt(0.04) = 0.2000000 $
   - *Noisy State Output ($x_1$):*
     $ x_1 = sqrt(1 - beta_1) x_0 + sqrt(beta_1) \epsilon = 1.9595918 + 0.2000000 \times 0.5 = 1.9595918 + 0.1000000 = 2.0595918 $

3. *Stochastic Expectation & Variance Preservation Interpretation:*
   - *Stochastic Expectation:* Under the conditional forward distribution $q(x_1 | x_0) = cal(N)(x_1; sqrt(1 - beta_1) x_0, beta_1)$, the expected value of $x_1$ given $x_0$ is:
     $ bb(E)[x_1 | x_0] = sqrt(1 - beta_1) x_0 = 1.9595918 $
     This contraction factor $sqrt(1 - \beta_1) \approx 0.9798$ slightly scales down the signal magnitude at each timestep.
   - *Variance Preservation:* If the data distribution $x_0$ has unit variance ($	ext{Var}(x_0) = 1$) and noise $\epsilon \sim cal(N)(0, 1)$ is independent of $x_0$, the total marginal variance of $x_1$ is:
     $ 	ext{Var}(x_1) = (sqrt(1 - beta_1))^2 	ext{Var}(x_0) + (sqrt(beta_1))^2 	ext{Var}(\epsilon) = (1 - 0.04)(1) + 0.04(1) = 0.96 + 0.04 = 1.0 $
     This property confirms that the variance schedule preserves total unit variance throughout all forward timesteps, preventing signal explosion or collapse.

=== Part B — Step-by-Step Loss Calculation for $L_{\text{simple}}$ (Section III)

Following the forward noising calculation in Part A, we demonstrate a single-sample gradient step evaluation for the simplified loss function $L_{\text{simple}}$.

1. *Neural Noise Prediction & Loss Computation:*
   - *Ground Truth Injected Noise:* $\epsilon = 0.5000$.
   - *U-Net Prediction:* Assume the neural network $\epsilon_theta(x_1, t=1)$ receives the noisy state $x_1 = 2.0595918$ and timestep $t=1$, predicting a noise scalar value of:
     $ \epsilon_theta(x_1, 1) = 0.4200 $
   - *Residual Noise Error:*
     $ \Delta \epsilon = \epsilon - \epsilon_theta(x_1, 1) = 0.5000 - 0.4200 = 0.0800 $
   - *Squared Error Loss for Sample ($L_{\text{simple}}^{(1)}$):*
     $ L_{\text{simple}}^{(1)} = \norm{\epsilon - \epsilon_theta(x_1, 1)}^2 = (0.0800)^2 = 0.0064 $

2. *Optimization & Gradient Interpretation:*
   - The computed squared error $0.0064$ contributes directly to the stochastic objective $L_{\text{simple}} = bb(E)_(t, x_0, \epsilon) \left[ \norm{\epsilon - \epsilon_theta(x_t, t)}^2 \right]$.
   - Backpropagating this error calculates the gradient $\nabla_theta L_{\text{simple}}^{(1)} = -2 (\epsilon - \epsilon_theta(x_1, 1)) \nabla_theta \epsilon_theta(x_1, 1)$, adjusting neural network weights $\theta$ so that future predictions $\epsilon_theta(x_1, 1)$ shift closer to the ground truth noise $\epsilon = 0.5000$.

== Latent Diffusion Models (LDM)

=== 1. VAE Encoder / Decoder & Perceptual Compression

To overcome the severe computational bottlenecks of pixel-space diffusion models, Latent Diffusion Models (LDM) introduce a explicit two-stage architecture that decouples perceptual compression from generative semantic modeling.

1. *Perceptual Spatial Compression ($z = cal(E)(x)$):*
   An autoencoder consisting of an encoder $cal(E)$ compresses an RGB pixel image $x \in \bb(R)^(H \times W \times 3)$ into a lower-dimensional latent representation $z \in \bb(R)^(h \times w \times c)$:
   $ z = cal(E)(x) $
   The spatial dimensions are downsampled by a factor $f = H / h = W / w = 2^m$ (where $m \in \bb(N)$). LDM explores downsampling factors $f \in \{1, 2, 4, 8, 16, 32\}$. For instance, an $f = 8$ compression factor reduces a $256 \times 256 \times 3$ pixel image to a $32 \times 32 \times 4$ latent tensor, discarding high-frequency imperceptible details while preserving essential structural semantics.

2. *Perceptual Reconstruction ($tilde(x) = cal(D)(z_0)$):*
   A decoder $cal(D)$ learns to map the denoised latent representation $z_0$ back to the original image space:
   $ tilde(x) = cal(D)(z_0) = cal(D)(cal(E)(x)) $
   To prevent high-variance latent spaces during autoencoder training, two regularization variants are utilized:
   - *KL-reg:* Imposes a mild KL-penalty towards a standard normal distribution on the learned latent space, similar to a Variational Autoencoder.
   - *VQ-reg:* Incorporates a vector quantization layer within the decoder, acting as a VQ-GAN latent space.
   Combined with perceptual loss terms (LPIPS) and patch-based adversarial losses, the autoencoder ensures that $tilde(x)$ maintains high visual fidelity without blurriness.

=== 2. Diffusion Operations in Latent Space & Computational Efficiency

By shifting the generative diffusion process into the lower-dimensional latent space $\bb(R)^(h \times w \times c)$, LDM executes all forward noising and reverse denoising transitions on latent vectors $z_t$.

1. *Forward Latent Diffusion:*
   $ q(z_1:T | z_0) = \prod_{t=1}^T q(z_t | z_(t-1)), \quad q(z_t | z_0) = cal(N)(z_t; sqrt(overline(\alpha)_t) z_0, (1 - overline(\alpha)_t) bold(I)) $
   Arbitrary latent state sampling follows the closed form $z_t(z_0, \epsilon) = sqrt(overline(\alpha)_t) z_0 + sqrt(1 - overline(\alpha)_t) \epsilon$, where $\epsilon \sim cal(N)(bold(0), bold(I))$.

2. *Reverse Latent Diffusion & Objective Function:*
   The U-Net neural network $\epsilon_theta(z_t, t, tau_theta(y))$ predicts the noise vector $\epsilon$ injected into the latent space:
   $ L_{\text{LDM}} = bb(E)_(cal(E)(x), y, \epsilon, t) \left[ \norm{\epsilon - \epsilon_theta(z_t, t, tau_theta(y))}^2 \right] $ <eq:lldm>

3. *Substantial Computational Savings:*
   - *Spatial Token Reduction:* Compressing image resolution by factor $f$ reduces spatial dimension size by $f^2$. For $f = 8$, the spatial token sequence length drops by a factor of $64$ (e.g., from $256^2 = 65,536$ pixels to $32^2 = 1,024$ latent tokens).
   - *Resource Efficiency:* Operating on $z_t$ lowers memory footprint and computational FLOPs for UNet self-attention blocks, accelerating training convergence and boosting sampling throughput by $3\times$ to $10\times$ compared to pixel-space DDPM.

=== 3. Cross-Attention Conditioning Mechanism

To transform diffusion models into versatile, multi-modal conditional generators, LDM integrates a flexible Cross-Attention mechanism into the U-Net backbone.

1. *Domain-Specific Encoder ($tau_theta(y)$):*
   Conditioning inputs $y$ (such as text prompts, layout bounding boxes, semantic segmentation maps, or target images) are processed by a specialized domain encoder $tau_theta(y) \in \bb(R)^(M \times d_tau)$ that maps $y$ into an intermediate representation vector of sequence length $M$ and dimension $d_tau$ (e.g., a Transformer/BERT encoder for text).

2. *Cross-Attention Formulation:*
   Let $\phi_i(z_t) \in \bb(R)^(N \times d_\epsilon^i)$ denote the flattened intermediate spatial feature map of the U-Net at layer $i$, where $N = h_i \times w_i$. The cross-attention layer projects features into Query, Key, and Value matrices:
   $ Q = W_Q^{(i)} \cdot \phi_i(z_t) \in \bb(R)^(N \times d), \quad K = W_K^{(i)} \cdot tau_theta(y) \in \bb(R)^(M \times d), \quad V = W_V^{(i)} \cdot tau_theta(y) \in \bb(R)^(M \times d) $
   The attention weights and output feature updates are computed as:
   $ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{sqrt(d)} \right) \cdot V $ <eq:crossattn>
   where $W_Q^{(i)} \in \bb(R)^(d \times d_\epsilon^i)$, $W_K^{(i)} \in \bb(R)^(d \times d_tau)$, and $W_V^{(i)} \in \bb(R)^(d \times d_tau)$ are learnable projection weight matrices, and $d$ is the projection channel dimension.

3. *Multi-Modal Interaction:*
   The matrix product $Q K^T \in \bb(R)^(N \times M)$ evaluates pairwise affinity between each spatial location in the U-Net ($N$) and each conditioning token ($M$), enabling precise spatial alignment and semantic control (e.g., placing objects described in text at specific spatial regions).

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
              *Pixel Image $x$*              $H \times W \times 3$
            ]
          ],
          [
            #text(size: 10pt, fill: rgb("2b7bba"))[$-->$]            #text(size: 6.5pt, fill: rgb("2b7bba"))[Encoder $cal(E)$]
          ],
          [
            #rect(fill: rgb("fff8e7"), stroke: 0.8pt + rgb("d35400"), radius: 3pt, inset: 5pt)[
              *Latent Code $z_0$*              $h \times w \times c$              #text(size: 6.5pt, fill: luma(100))[(Factor $f = H/h$)]
            ]
          ],
          [
            #text(size: 10pt, fill: rgb("27ae60"))[$-->$]            #text(size: 6.5pt, fill: rgb("27ae60"))[Decoder $cal(D)$]
          ],
          [
            #rect(fill: rgb("e8f8f5"), stroke: 0.8pt + rgb("27ae60"), radius: 3pt, inset: 5pt)[
              *Reconstructed $tilde(x)$*              $H \times W \times 3$
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
              *Latent $z_0$*              $cal(E)(x)$
            ]
          ],
          [
            #text(size: 9pt, fill: rgb("c0392b"))[$<-->$]            #text(size: 6.5pt, fill: rgb("c0392b"))[Diffusion $q / p_theta$]
          ],
          [
            #rect(fill: rgb("f3e5f5"), stroke: 0.8pt + rgb("8e44ad"), radius: 3pt, inset: 6pt)[
              *Denoising U-Net $\epsilon_theta(z_t, t, tau_theta(y))$*              #v(2pt)
              #rect(fill: rgb("ffffff"), stroke: 0.5pt + rgb("8e44ad"), radius: 2pt, inset: 3pt)[
                *Cross-Attention Layer*                $Q = W_Q \phi_i(z_t), \ K = W_K tau_theta(y), \ V = W_V tau_theta(y)$                $\text{Attention}(Q,K,V) = \text{softmax}(Q K^T / sqrt(d)) V$
              ]
            ]
          ],
          [
            #text(size: 9pt, fill: rgb("8e44ad"))[$<--$]            #text(size: 6.5pt, fill: rgb("8e44ad"))[$K, V$ Tokens]
          ],
          [
            #rect(fill: rgb("fef9e7"), stroke: 0.8pt + rgb("f39c12"), radius: 3pt, inset: 5pt)[
              *Conditioning $y$*              Text / Layout / Maps              #v(2pt)
              #text(size: 6.5pt)[Encoder $tau_theta(y)$]
            ]
          ]
        )
      ]
    )
    #v(4pt)
    #align(center)[
      #text(size: 7.5pt, style: "italic", fill: luma(80))[
        Full LDM Pipeline: Image $x$ is compressed into latent space $z_0 = cal(E)(x)$. Diffusion runs in latent space on $z_t$. Conditioning $y$ is encoded by $tau_theta(y)$ and injected via Cross-Attention layers inside the UNet $\epsilon_theta$. Denoised $z_0$ is decoded back to pixel space $tilde(x) = cal(D)(z_0)$.
      ]
    ]
  ]
] <fig:ldm-pipeline>
