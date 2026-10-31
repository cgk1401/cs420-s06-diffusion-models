# Technical Report: High-Resolution Image Synthesis with Latent Diffusion Models (LDM)

**Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  
**Affiliations:** Ludwig Maximilian University of Munich, IWR Heidelberg University, Runway ML  
**Conference:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)  
**Project/Codebase:** [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) (Foundational architecture of **Stable Diffusion**)

---

## 1. Executive Summary & Problem Formulation

### 1.1. The Computational Bottleneck of Pixel-Space Diffusion
Denoising Diffusion Probabilistic Models (DDPMs) and Score-Based Generative Models achieve high visual fidelity and mode coverage, surpassing Generative Adversarial Networks (GANs) without suffering from training instabilities or mode collapse. However, standard diffusion models operate directly in **pixel space** ($\mathbb{R}^{H \times W \times 3}$). This introduces severe computational drawbacks:

1. **Prohibitive Training Cost:** Optimizing deep convolutional networks on high-dimensional pixel arrays over thousands of diffusion timesteps consumes hundreds to thousands of GPU days (e.g., 150–1000 V100 days for ADM).
2. **High Inference Latency:** Sampling requires repeatedly evaluating the model backbone through $T$ sequential denoising steps (typically $T \in [50, 1000]$), rendering sampling impractical on consumer hardware.

### 1.2. Perceptual vs. Semantic Compression
Rate-distortion analysis of likelihood-based generative models reveals two distinct regimes:

$$\text{Image Data} \xrightarrow{\text{Perceptual Compression}} \text{Compact Semantic Space} \xrightarrow{\text{Semantic Compression}} \text{Generative Model}$$

* **Perceptual Compression (High-Frequency Details):** Digital images contain imperceptible, high-frequency spatial variations. In pixel-based diffusion, gradient computations and model capacity are wasted on modeling subtle noise and micro-textures.
* **Semantic Compression (Conceptual Composition):** The generative model should focus on global scene structure, semantic layout, and object identities.

**Core Proposal:** Explicitly separate the compression phase from the generative phase by training diffusion models within the lower-dimensional latent space of a pretrained, robust autoencoder. The resulting framework is called **Latent Diffusion Models (LDMs)**.

---

## 2. Theoretical Architecture of Latent Diffusion Models

The LDM framework decouples generative modeling into two distinct components:
1. An **Autoencoding Engine** ($\mathcal{E}, \mathcal{D}$) for perceptual compression.
2. A **Denoising U-Net** ($\epsilon_\theta$) for latent diffusion and semantic generation.

```
+-----------------------------------------------------------------------------------------+
|                                    LATENT SPACE                                         |
|                                                                                         |
|      +---------------+       Forward Noise        +---------------+                     |
|      |  Latent z_0   | -------------------------> |  Latent z_t   |                     |
|      +---------------+                            +---------------+                     |
|              ^                                            |                             |
|              |         Reverse Denoising U-Net            |                             |
|              +--------------------------------------------+                             |
|                                     ^                                                   |
|                                     | Cross-Attention                                   |
|                             +---------------+                                           |
|                             |  Encoder tau  | <=== Conditioning y (Text, Bounding Box)  |
|                             +---------------+                                           |
+-----------------------------------------------------------------------------------------+
       ^                                                    |
       | Encoder E                                          | Decoder D
+---------------+                                    +---------------+
| Input Image x |                                    | Output Image x|
|  (Pixel RGB)  |                                    |  (Pixel RGB)  |
+---------------+                                    +---------------+
```

---

### 2.1. First Stage: Perceptual Image Compression

Given an image $x \in \mathbb{R}^{H \times W \times 3}$, an encoder $\mathcal{E}$ maps $x$ to a spatial latent code:

$$z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$$

where $f = H/h = W/w = 2^m$ represents the spatial downsampling factor ($m \in \mathbb{N}$). A decoder $\mathcal{D}$ reconstructs the image from the latent space:

$$\tilde{x} = \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x)) \approx x$$

#### Loss Formulation
To avoid blurry reconstructions typically produced by standard $\mathcal{L}_1$ or $\mathcal{L}_2$ pixel-regression losses, the autoencoder is trained using a composite perceptual and adversarial objective:
* **LPIPS Perceptual Loss:** Evaluated across intermediate feature activations of a pretrained VGG network to preserve semantic details.
* **PatchGAN Adversarial Loss:** Enforces local realism and crisp textures on image patches.

#### Latent Regularization Schemes
To prevent arbitrarily scaled or unbounded latent spaces, two regularizations were evaluated:
1. **KL-Regularization (KL-reg):** Imposes a mild Kullback-Leibler penalty toward a standard normal distribution $\mathcal{N}(0, I)$, analogous to a Variational Autoencoder (VAE).
2. **Vector Quantization (VQ-reg):** Employs a vector quantization codebook layer within the decoder, similar to VQGAN, but absorbs the quantization stage directly into the autoencoding process.

Because the subsequent diffusion model retains the 2D spatial arrangement of $z$, mild downsampling rates ($f \in \{4, 8\}$) preserve spatial inductive biases without requiring 1D sequence flattening.

---

### 2.2. Second Stage: Latent Diffusion

Standard diffusion models optimize an unweighted variational bound on negative log-likelihood:

$$L_{\text{DM}} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0, I), t} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|_2^2 \right]$$

In LDMs, the forward diffusion process is applied directly to the latent code $z = \mathcal{E}(x)$. The closed-form noisy latent $z_t$ at arbitrary timestep $t \in \{1, \dots, T\}$ is given by:

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

The neural network backbone $\epsilon_\theta(z_t, t)$ is trained via the latent objective:

$$L_{\text{LDM}} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0, I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|_2^2 \right]$$

**Advantages of Latent-Space Diffusion:**
* **Dimensionality Reduction:** An image of size $512 \times 512 \times 3$ mapped with downsampling factor $f = 8$ yields a latent tensor of size $64 \times 64 \times 4$. The spatial area is reduced by a factor of $f^2 = 64$.
* **Computational Efficiency:** Flops per denoising step drop significantly, enabling higher batch sizes and faster training convergence.
* **Separation of Concerns:** The autoencoder handles pixel-level details, leaving the diffusion model to focus on global composition and semantics.

---

### 2.3. Conditioning Mechanisms via Cross-Attention

To handle diverse multi-modal conditioning inputs (text prompts, semantic segmentation maps, bounding boxes, or low-resolution images), the authors augmented the U-Net architecture with **cross-attention mechanisms**.

```
Intermediate U-Net Features φ_i(z_t) ────> Q = W_Q^(i) · φ_i(z_t)
                                                  │
                                                  ▼
Conditioning Input y ──> [ Encoder τ_θ ] ──> K = W_K^(i) · τ_θ(y) ──> Softmax( Q K^T / √d ) · V
                                         ──> V = W_V^(i) · τ_θ(y)
```

1. **Domain-Specific Encoders ($\tau_\theta$):**
   * Text inputs: Pretrained Transformer / BERT / CLIP text encoders.
   * Semantic layouts / class labels: Discrete embedding lookup or MLP projections.
   * Spatially aligned conditions (depth, inpainting masks): Concatenated directly along the channel dimension.

2. **Cross-Attention Formulation:**  
   Given flattened intermediate U-Net features $\varphi_i(z_t) \in \mathbb{R}^{N \times d_\epsilon}$ and conditioning representation $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$:

   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

   where:

   $$Q = W_Q^{(i)} \cdot \varphi_i(z_t), \quad K = W_K^{(i)} \cdot \tau_\theta(y), \quad V = W_V^{(i)} \cdot \tau_\theta(y)$$

   with learnable projection weights $W_Q^{(i)} \in \mathbb{R}^{d \times d_\epsilon}$, $W_K^{(i)} \in \mathbb{R}^{d \times d_\tau}$, and $W_V^{(i)} \in \mathbb{R}^{d \times d_\tau}$.

3. **Joint Conditional Training Objective:**

   $$L_{\text{CLDM}} := \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0, I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2 \right]$$

---

## 3. Systematic Analysis of Downsampling Factors ($f$)

The paper investigated downsampling factors $f \in \{1, 2, 4, 8, 16, 32\}$ on class-conditional ImageNet under identical compute budgets (single NVIDIA A100 GPU):

| Downsampling Factor | Spatial Latent Dim ($256^2$ input) | Training Throughput | Sample Quality & Visual Fidelity |
| :--- | :--- | :--- | :--- |
| **LDM-1** (Pixel Space) | $256 \times 256$ | Very Slow (low sample/sec) | Poor FID under fixed compute budget |
| **LDM-2** | $128 \times 128$ | Slow | Suboptimal convergence |
| **LDM-4** | $64 \times 64$ | Fast | **Optimal balance** for complex distributions |
| **LDM-8** | $32 \times 32$ | Very Fast | **Optimal balance**; lowest FID on ImageNet |
| **LDM-16** | $16 \times 16$ | High | Information loss begins in first stage |
| **LDM-32** | $8 \times 8$ | Maximum | Stagnant FID; first-stage bottleneck |

```
              FID Score vs. Downsampling Factor f
              
    High FID ^
    (Worse)  |      LDM-1 (Pixel Space - Compute Bound)
             |        \
             |         \
             |          \                 LDM-32 (Information Bottleneck)
             |           \                      /
             |            LDM-2                /
             |              \                 /
             |               \--- LDM-4 --- LDM-8 (Sweet Spot)
    Low FID  +---------------------------------------------------->
    (Better) 1            2       4         8       16       32
                               Downsampling Factor (f)
```

**Key Takeaways:**
* Small $f$ ($f=1, 2$) leaves perceptual compression to the diffusion model, wasting capacity on imperceptible noise.
* High $f$ ($f=32$) creates an excessively lossy autoencoder bottleneck, setting an upper bound on output quality regardless of diffusion training.
* **LDM-4 and LDM-8 represent the Pareto-optimal regime**, yielding the lowest FID scores while accelerating sampling throughput.

---

## 4. Experimental Evaluations and Results

### 4.1. Unconditional Image Synthesis ($256 \times 256$)

Evaluated using Fréchet Inception Distance (FID) and Precision/Recall:

| Dataset | Model | FID $\downarrow$ | Precision $\uparrow$ | Recall $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- |
| **CelebA-HQ** | StyleGAN2 | 5.91 | 0.72 | 0.45 |
| | LSGM | 7.22 | - | - |
| | **LDM-4 (Ours)** | **5.11** | **0.72** | **0.49** |
| **FFHQ** | StyleGAN | 4.16 | 0.71 | 0.46 |
| | **LDM-4 (Ours)** | **4.98** | **0.73** | **0.50** |
| **LSUN-Churches** | DDPM | 7.89 | - | - |
| | StyleGAN2 | 3.86 | - | - |
| | **LDM-8 (Ours)** | **4.02** | **0.64** | **0.52** |
| **LSUN-Bedrooms** | ADM | 1.90 | 0.66 | 0.51 |
| | **LDM-4 (Ours)** | **2.95** | **0.66** | **0.48** |

*Note: On LSUN-Bedrooms, LDM achieved performance close to ADM while utilizing half the parameters and requiring $4\times$ fewer training resources.*

---

### 4.2. Text-Conditional Synthesis (MS-COCO Validation)

Trained with a 1.45B parameter backbone on LAION-400M using a Transformer text encoder:

| Model | Paradigm | FID $\downarrow$ | Inception Score (IS) $\uparrow$ |
| :--- | :--- | :--- | :--- |
| **DALL-E** | Autoregressive (Discrete Latent) | 27.50 | 17.90 |
| **CogView** | Autoregressive (Discrete Latent) | 27.10 | 18.20 |
| **LAFITE** | GAN | 26.94 | 26.02 |
| **LDM-KL-8** | Latent Diffusion | 23.35 | 19.93 |
| **LDM-KL-8-G** | Latent Diffusion + CFG ($s=1.5$) | **12.61** | **26.62** |

*Note: Incorporating Classifier-Free Guidance (CFG) reduced the FID from 23.35 to 12.61, outperforming competitive autoregressive and GAN models.*

---

### 4.3. Image Inpainting and Object Removal

Evaluated on $512 \times 512$ crops from the Places dataset:
* **Concatenation Conditioning:** The latent representations of the unmasked context and the downsampled binary mask are concatenated channel-wise to the noisy latent $z_t$.
* **Performance:** LDM achieved an FID of **1.50** on all test samples and **9.39** on severe occlusions ($40\text{--}50\%$ masked area), outperforming specialized baselines (such as LaMa and CoModGAN) while generating diverse, semantically coherent completions.

---

### 4.4. Super-Resolution ($64 \rightarrow 256$)

* **Architecture:** Conditioned on the bicubic low-resolution input via channel concatenation.
* **Results:** LDM-SR achieved an FID of **2.13** on ImageNet validation, outperforming SR3 (FID 5.30). User studies indicated a strong human preference for LDM-SR over pure regression and pixel-diffusion baselines.

---

### 4.5. Convolutional Sampling Beyond $256^2$

Because the U-Net operates convolutionally over the latent space, an LDM trained on $256 \times 256$ images can be applied to generate spatially coherent megapixel scenes ($512 \times 1024$) at inference time. This capability was demonstrated on semantic landscape synthesis by evaluating the model across extended spatial grids in a single pass.

---

## 5. Architectural & Technical Contributions Summary

1. **Separation of Perceptual and Semantic Learning:** Decouples high-frequency compression (via an adversarial/perceptual autoencoder) from generative modeling (via latent diffusion).
2. **Convolutional Inductive Bias Preservation:** Retains the 2D spatial structure in the latent representation, avoiding the quadratic sequence-length scaling issues seen in autoregressive vision transformers.
3. **Cross-Attention Conditioning Engine:** Establishes a flexible conditioning mechanism for text prompts, semantic layouts, and multi-modal signals without requiring architectural redesigns.
4. **Computational Accessibility:** Achieved state-of-the-art visual fidelity while reducing training and inference costs, enabling diffusion research on standard hardware.

---

## 6. Long-Term Impact

| Dimension | Pixel-Based Diffusion (DDPM/ADM) | Latent Diffusion (LDM / Stable Diffusion) |
| :--- | :--- | :--- |
| **Operating Space** | $\mathbb{R}^{H \times W \times 3}$ (Pixel array) | $\mathbb{R}^{h \times w \times c}$ (Latent space, e.g., $f=8$) |
| **Training Resource Footprint** | Hundreds of GPU days (Cluster scale) | Single GPU / Moderate cluster |
| **Inference Latency** | High (seconds/minutes per image) | Low (sub-second on consumer GPUs with DDIM) |
| **Conditioning Adaptability** | Primarily labels / low-res images | Multi-modal via Cross-Attention (Text, Layouts, Audio) |
| **Downstream Foundation** | Specialized research models | **Stable Diffusion v1/v2/XL, ControlNet, Runway Gen-1** |