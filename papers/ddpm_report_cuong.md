# Comprehensive Technical Report: Denoising Diffusion Probabilistic Models (DDPM)

**Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)  
**Conference:** NeurIPS 2020  
**Core Focus:** The Forward-Noising Process, Reverse-Denoising Process, and the Noise-Prediction Objective ($L_{\text{simple}}$).

---

## 1. Executive Summary & Historical Context

Prior to this work, deep generative modeling was predominantly driven by three paradigms:
* **Generative Adversarial Networks (GANs):** High sample fidelity, but notoriously unstable to optimize, susceptible to mode collapse, and lacking an explicit density formulation.
* **Variational Autoencoders (VAEs):** Principled optimization via the Evidence Lower Bound (ELBO), but prone to blurry samples when optimized under standard $\mathcal{L}_2$ pixel loss.
* **Autoregressive & Normalizing Flow Models:** Tractable exact likelihood computation, but restricted by severe architectural constraints or high computational overhead during generation.

Diffusion probabilistic models were originally introduced by Sohl-Dickstein et al. (2015), grounded in non-equilibrium thermodynamics. However, it was **Ho et al. (2020)** who turned diffusion models into state-of-the-art generative models. By establishing a formal mathematical bridge between **diffusion models**, **denoising score matching**, and **annealed Langevin dynamics**, DDPM matched and exceeded GANs in sample quality (achieving an FID of 3.17 on unconditional CIFAR-10).

---

## 2. The Forward-Noising Process ($q$)

The forward process (or diffusion process) is a parameter-free, fixed Markov chain that systematically destroys structure in a data point $x_0 \sim q(x_0)$ by injecting Gaussian noise over $T$ discrete timesteps ($t = 1, \dots, T$).

```
x_0 ----> x_1 ----> x_2 ----> ... ----> x_{t-1} ----> x_t ----> ... ----> x_T ~ N(0, I)
      q         q                                  q
```

### 2.1. Step-Wise Transition Kernel

At each timestep $t$, Gaussian noise is added according to a predefined variance schedule $\beta_1, \dots, \beta_T$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right)
$$

Where:
* $\beta_t \in (0, 1)$ is held constant as a non-learnable hyperparameter, linearly scaled from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.
* When $T = 1000$ is sufficiently large, the original signal in $x_0$ is completely degraded, yielding:

$$
p(x_T) \approx \mathcal{N}(0, I)
$$

The joint distribution of the complete forward trajectory conditioned on $x_0$ is expressed as:

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

---

### 2.2. Closed-Form Sampling at Arbitrary Timestep $t$

A foundational property of the Gaussian forward process is that an intermediate latent $x_t$ can be sampled in closed form directly from $x_0$ without iteratively evaluating intermediate steps $x_1, \dots, x_{t-1}$.

Let $\alpha_t = 1 - \beta_t$ and define the cumulative product:

$$
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

Using the reparameterization trick, we express the step transition as:

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1}, \quad \text{where } \epsilon_{t-1} \sim \mathcal{N}(0, I)
$$

Expanding this relation recursively:

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2} \right) + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1}
\end{aligned}
$$

Because the sum of two independent Gaussians $\mathcal{N}(0, \sigma_a^2 I)$ and $\mathcal{N}(0, \sigma_b^2 I)$ is $\mathcal{N}(0, (\sigma_a^2 + \sigma_b^2) I)$, the merged variance is:

$$
\alpha_t (1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}
$$

Induction through all timesteps down to $x_0$ produces the conditional marginal distribution:

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I\right)
$$

Equivalently, using standard Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$:

$$
x_t(x_0, \epsilon) = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

This closed-form formulation allows $\mathcal{O}(1)$ sampling of noisy states at training time by sampling $t \sim \text{Uniform}(\{1, \dots, T\})$ uniformly at random.

---

### 2.3. Tractable Forward Posterior $q(x_{t-1} \mid x_t, x_0)$

While the unconditional reverse step $q(x_{t-1} \mid x_t)$ is intractable (requiring integration over the entire data distribution $q(x_0)$), conditioning on $x_0$ renders the posterior tractable via Bayes' rule:

$$
q(x_{t-1} \mid x_t, x_0) = q(x_t \mid x_{t-1}, x_0) \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} = q(x_t \mid x_{t-1}) \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

Since all distributions on the right-hand side are Gaussian, $q(x_{t-1} \mid x_t, x_0)$ is also Gaussian:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\right)
$$

Matching quadratic coefficients in the exponential yields:

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

This tractable posterior serves as the analytical ground truth that the reverse process neural network is trained to approximate.

---

## 3. The Reverse-Denoising Process ($p_\theta$)

The generative process is modeled as a parameterized Markov chain that starts from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and sequentially denoises the latent representation back to data space:

```
x_T ----> x_{T-1} ----> ... ----> x_t ----> x_{t-1} ----> ... ----> x_0
      p_\theta                       p_\theta                    p_\theta
```

The joint distribution is defined as:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
$$

where each reverse transition is parameterized by a neural network:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\right)
$$

### Parameterization of Covariance $\Sigma_\theta(x_t, t)$

The authors investigated parameterizing $\Sigma_\theta(x_t, t)$ as a learned diagonal covariance matrix, but discovered that training was unstable. Instead, they fixed the covariance to time-dependent scalar variances:

$$
\Sigma_\theta(x_t, t) = \sigma_t^2 I
$$

Two analytical boundary options were compared:
1. $\sigma_t^2 = \beta_t$ (optimal when $x_0 \sim \mathcal{N}(0, I)$).
2. $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ (optimal when $x_0$ is a deterministic delta function).

Empirically, both options yielded comparable sample quality. Fixing $\sigma_t^2$ leaves the network focused entirely on predicting the mean $\mu_\theta(x_t, t)$.

---

## 4. The Noise-Prediction Objective

### 4.1. ELBO Decomposition into KL Divergences

The model parameters $\theta$ are trained to minimize the negative log-likelihood via the Variational Upper Bound $L \ge \mathbb{E}[-\log p_\theta(x_0)]$:

$$
L = \mathbb{E}_q \left[ -\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]
$$

Using Bayes' theorem and the Markov property, this bound is decomposed into low-variance, closed-form KL divergence terms:

$$
L = \mathbb{E}_q \left[ \underbrace{D_{\text{KL}}(q(x_T \mid x_0) \parallel p(x_T))}_{L_T} + \sum_{t > 1} \underbrace{D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0} \right]
$$

Component breakdown:
* **$L_T$:** Measures how close the final diffused state $q(x_T \mid x_0)$ is to the prior $\mathcal{N}(0, I)$. Because $\beta_t$ is fixed, $L_T$ contains no trainable parameters ($\theta$) and evaluates to $\approx 10^{-5}$ bits/dim, making it negligible during training.
* **$L_0$:** The reconstruction term from $x_1$ to data space $x_0$, evaluated using a discretized continuous Gaussian decoder across discrete image values in $[0, 255]$ mapped to $[-1, 1]$.
* **$L_{t-1}$ ($t > 1$):** Directly compares the true posterior $q(x_{t-1} \mid x_t, x_0)$ with the model transition $p_\theta(x_{t-1} \mid x_t)$.

Because both distributions are isotropic Gaussians, their KL divergence reduces to an analytical Mean Squared Error (MSE) between their mean vectors:

$$
L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2\sigma_t^2} \left\Vert \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\Vert^2 \right] + C
$$

where $C$ is a constant independent of $\theta$.

---

### 4.2. Reparameterization from Mean Prediction to Noise Prediction

From the closed-form forward sampling formula:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \implies x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)
$$

Substituting this formulation of $x_0$ into the analytical mean $\tilde{\mu}_t(x_t, x_0)$:

$$
\begin{aligned}
\tilde{\mu}_t(x_t, x_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}} \right) + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t \\
&= \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)
\end{aligned}
$$

This identity illustrates that the true mean $\tilde{\mu}_t$ is completely specified by $x_t$ and the noise vector $\epsilon$. Because $x_t$ is already available to the network as an input, the model mean $\mu_\theta$ can be reparameterized to predict the injected noise $\epsilon$:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

where $\epsilon_\theta$ is a neural network approximating $\epsilon$ from $(x_t, t)$.

Substituting this parameterization into $L_{t-1}$ yields:

$$
L_{t-1} - C = \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\Vert \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t\right) \right\Vert^2 \right]
$$

---

### 4.3. The Simplified Objective ($L_{\text{simple}}$)

The rigorous derivation above includes the scalar weight:

$$
w_t = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}
$$

The primary algorithmic innovation in DDPM was **dropping this weighting entirely**. Ho et al. introduced the unweighted objective:

$$
L_{\text{simple}}(\theta) := \mathbb{E}_{t, x_0, \epsilon} \left[ \left\Vert \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t\right) \right\Vert^2 \right]
$$

where $t \sim \text{Uniform}(\{1, \dots, T\})$ and $\epsilon \sim \mathcal{N}(0, I)$.

#### Why Dropping $w_t$ Enhances Sample Fidelity

1. **Downweighting Microscopic Noise ($t \to 1$):** Under the exact ELBO, $w_t$ is exceedingly large at small $t$, where the signal-to-noise ratio is high. This forces the network to allocate capacity to imperceptible, high-frequency details (which dominate bits/dim metrics but contribute little to perceptual quality).
2. **Prioritizing Macro-Structure ($t \gg 1$):** By setting the effective loss weight to $1$, $L_{\text{simple}}$ acts as a reweighted variational bound that emphasizes large $t$ terms. At large $t$, the network learns global scene layout, coarse shapes, and semantic structure, which drives dramatic improvements in FID and Inception Score.

---

### 4.4. Connections to Score Matching and Langevin Dynamics

The noise-prediction parameterization reveals an exact equivalence with **Denoising Score Matching** (Vincent, 2011; Song & Ermon, 2019).

The score of a continuous data distribution is defined as $\nabla_{x_t} \log q(x_t)$. For the perturbed marginal $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)$:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = - \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = - \frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}
$$

Thus:

$$
\epsilon_\theta(x_t, t) \propto - \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log q(x_t)
$$

The network $\epsilon_\theta$ directly estimates the scaled score function of the diffused data distribution. Sampling from $p_\theta(x_{t-1} \mid x_t)$ is mathematically equivalent to running **Annealed Langevin Dynamics**, progressively steering noisy vectors along the estimated gradient of the data manifold.

---

## 5. Training and Sampling Algorithms

```
================================================================================
Algorithm 1: Training
================================================================================
1: repeat
2:    x_0 ~ q(x_0)                                   // Sample training image
3:    t ~ Uniform({1, ..., T})                       // Sample random timestep
4:    \epsilon ~ N(0, I)                             // Sample standard Gaussian noise
5:    Take gradient descent step on:
         \nabla_\theta || \epsilon - \epsilon_\theta( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t ) ||^2
6: until converged
================================================================================
```

```
================================================================================
Algorithm 2: Sampling (Generation)
================================================================================
1: x_T ~ N(0, I)                                     // Start from pure Gaussian noise
2: for t = T, ..., 1 do
3:    z ~ N(0, I) if t > 1, else z = 0
4:    x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
5: end for
6: return x_0
================================================================================
```

---

## 6. Architecture & Implementation Highlights

* **Backbone:** U-Net based on PixelCNN++ and Wide-ResNet, maintaining identical spatial resolutions between input $x_t$ and output $\epsilon_\theta$.
* **Time Conditioning:** Timestep $t$ is projected using Transformer sinusoidal positional embeddings, expanded through MLP layers, and added into each residual block.
* **Normalization:** Group Normalization is utilized across all convolutional layers.
* **Attention Mechanism:** Self-attention blocks are placed at the $16 \times 16$ intermediate feature resolution to capture non-local spatial dependencies.

---

## 7. Ablation Analysis: Parameterization & Objective

Empirical results on unconditional CIFAR-10 from the original paper:

| Target Parameterization | Mean Formulation | Covariance $\Sigma$ | Objective | FID $\downarrow$ | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- |
| True Variational Bound | Predict $\tilde{\mu}$ | Learned diagonal $\Sigma_\theta$ | Full ELBO ($L$) | 23.69 | Unstable optimization |
| True Variational Bound | Predict $\tilde{\mu}$ | Fixed $\sigma_t^2$ | Full ELBO ($L$) | 13.51 | Stable, moderate quality |
| True Variational Bound | Predict $\epsilon$ | Fixed $\sigma_t^2$ | Full ELBO ($L$) | 13.22 | Comparable to $\tilde{\mu}$ on $L$ |
| **Simplified Objective** | **Predict** $\epsilon$ | **Fixed** $\sigma_t^2$ | **$L_{\text{simple}}$** | **3.17** | **State-of-the-Art (NeurIPS 2020)** |

---

## 8. Key Takeaways & Significance

1. **Simplicity over Strict Likelihood:** By trading strict likelihood bounds for perceptual weighting ($L_{\text{simple}}$), DDPM drastically simplified generative training into an intuitive denoising task: predicting the Gaussian noise injected into an image.
2. **Coarse-to-Fine Progressive Generation:** The reverse trajectory generates macro-level features (semantics, layout) at high $t$, reserving small $t$ steps for subtle high-frequency details.
3. **Foundational Blueprint:** The formulation in this paper became the underlying mathematical engine for modern generative frameworks, including **Latent Diffusion Models (Stable Diffusion), DALL-E 2, Imagen, and Midjourney**.