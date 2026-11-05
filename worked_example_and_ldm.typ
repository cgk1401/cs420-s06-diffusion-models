= Worked Example & Latent Diffusion Models (LDM)

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
   - *Stochastic Expectation:* Under the conditional forward distribution $q(x_1 | x_0) = cal(N)(x_1; sqrt(1 - beta_1) x_0, beta_1 \mathbf{I})$, the expected value of $x_1$ given $x_0$ is:
     $ bb(E)[x_1 | x_0] = sqrt(1 - beta_1) x_0 = 1.9595918 $
     This contraction factor $sqrt(1 - \beta_1) \approx 0.9798$ slightly scales down the signal magnitude at each timestep.
   - *Variance Preservation:* If the data distribution $x_0$ has unit variance ($\text{Var}(x_0) = 1$) and noise $\epsilon \sim cal(N)(0, 1)$ is independent of $x_0$, the total marginal variance of $x_1$ is:
     $ \text{Var}(x_1) = (sqrt(1 - beta_1))^2 \text{Var}(x_0) + (sqrt(beta_1))^2 \text{Var}(\epsilon) = (1 - 0.04)(1) + 0.04(1) = 0.96 + 0.04 = 1.0 $
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

To overcome the severe computational bottlenecks of pixel-space diffusion models, Latent Diffusion Models (LDM) introduce an explicit two-stage architecture that decouples perceptual compression from generative semantic modeling.

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
   $ q(z_{1:T} | z_0) = \prod_{t=1}^T q(z_t | z_(t-1)), \quad q(z_t | z_0) = cal(N)(z_t; sqrt(overline(\alpha)_t) z_0, (1 - overline(\alpha)_t) bold(I)) $
   Arbitrary latent state sampling follows the closed form $z_t(z_0, \epsilon) = sqrt(overline(\alpha)_t) z_0 + sqrt(1 - overline(\alpha)_t) \epsilon$, where $\epsilon \sim cal(N)(bold(0), bold(I))$.

2. *Reverse Latent Diffusion & Objective Function:*
   The U-Net neural network $\epsilon_theta(z_t, t, tau_theta(y))$ predicts the noise vector $\epsilon$ injected into the latent space:
   $ L_{\text{LDM}} = bb(E)_(cal(E)(x), y, \epsilon, t) \left[ \norm{\epsilon - \epsilon_theta(z_t, t, tau_theta(y))}^2 \right] $ <eq:lldm>

3. *Substantial Computational Savings:*
   - *Spatial Token Reduction:* Compressing image resolution by factor $f$ reduces spatial dimension size by $f^2$. For $f = 8$, the spatial token sequence length drops by a factor of $64$ (e.g., from $256^2 = 65,536$ pixels to $32^2 = 1,024$ latent tokens).
   - *Resource Efficiency:* Operating on $z_t$ lowers memory footprint and computational FLOPs for U-Net self-attention blocks, accelerating training convergence and boosting sampling throughput by $3\times$ to $10\times$ compared to pixel-space DDPM.

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
