# Step 0 — Verified 1D Worked Numerical Example (Section 3.2)

- **Dataset & Sample**: MNIST Test Set, Digit `3`, Resolution $28 \times 28$, Random Seed `42`
- **Selected Pixel Coordinate**: $(r, c) = (10, 14)$ on the bright stroke of digit `3`
- **Raw Grayscale Intensity**: $I_0 = 236 \in [0, 255]$
- **Normalized Clean Input ($x_0$)**:
  $$x_0 = \frac{236}{127.5} - 1 = 0.850980$$
- **Sampled Standard Gaussian Noise ($\epsilon \sim \mathcal{N}(0, 1)$ at pixel $(10, 14)$)**:
  $$\epsilon = -0.714962$$

## Part A: Single-Step 1D Forward Noising ($t = 1$)
- **Variance Schedule at $t = 1$**: $\beta_1 = 0.0001$, $\alpha_1 = 1 - \beta_1 = 0.9999$
- **Signal Scaling Factor**: $\sqrt{1 - \beta_1} = \sqrt{0.9999} = 0.999950$
- **Noise Standard Deviation**: $\sqrt{\beta_1} = \sqrt{0.0001} = 0.0100$
- **Conditional Mean ($\mu_1 = \mathbb{E}[x_1 \mid x_0]$)**:
  $$\mu_1 = \sqrt{1 - \beta_1}\, x_0 = 0.999950 \times 0.850980 = 0.850938$$
- **Stochastic Noise Perturbation ($\sqrt{\beta_1}\, \epsilon$)**:
  $$\sqrt{\beta_1}\, \epsilon = 0.0100 \times (-0.714962) = -0.007150$$
- **Resulting Noisy Pixel State ($x_1$)**:
  $$x_1 = \mu_1 + \sqrt{\beta_1}\, \epsilon = 0.850938 + (-0.007150) = 0.843788$$

## Part A (Extension): Closed-Form Jump to $t = 50$ at the Same Pixel $(10, 14)$
- **Cumulative Signal Product**: $\bar{\alpha}_{50} = 0.971016$
- **Signal Coefficient**: $\sqrt{\bar{\alpha}_{50}} = 0.985401$
- **Noise Coefficient**: $\sqrt{1 - \bar{\alpha}_{50}} = 0.170248$
- **Noisy State at $t = 50$**:
  $$x_{50} = 0.985401 \times (0.850980) + 0.170248 \times (-0.714962) = 0.716836$$

## Part B: Step-by-Step Loss Calculation ($L_{\text{simple}}$) at Pixel $(10, 14)$
- **True Injected Noise ($\epsilon$)**: $-0.714962$
- **U-Net Predicted Noise ($\epsilon_\theta(x_1, 1)$)**: $0.421943$
- **Prediction Residual ($\epsilon - \epsilon_\theta(x_1, 1)$)**:
  $$-0.714962 - (0.421943) = -1.136905$$
- **Pointwise Squared Error ($\|\epsilon - \epsilon_\theta(x_1, 1)\|^2$)**:
  $$(-1.136905)^2 = 1.292554$$
- **Full $28 \times 28$ Image MSE Loss at $t = 1$**: $1.134913$
