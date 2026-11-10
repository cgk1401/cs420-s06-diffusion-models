= Method and Worked Example
== DDPM: Forward Noising and Reverse Denoising

=== Definitions and Forward Process

We follow Ho et al. @ho2020ddpm (Sections 2-3). A clean image is a vector $x_0$ with $D = H times W times C$ coordinates, where $H$, $W$, and $C$ denote height, width, and channels. Integer intensities are scaled by $x_0 = x_"raw" / 127.5 - 1$ into $[-1, 1]^D$. This sets a consistent numerical range; it does not guarantee zero empirical mean or unit variance. The states after noising lie in $bb(R)^D$, so noisy coordinates need not remain in that interval. The principal symbols are defined in @tab:symbols.

#figure(
  kind: table,
  caption: [Notation used in the DDPM equations, following @ho2020ddpm. All scaled intensities, noise coordinates, and variances are dimensionless.],
  {
    set text(size: 9pt)
    set par(justify: false, first-line-indent: 0pt)
    table(
      columns: (1.1fr, 3fr), stroke: none, align: left, inset: 4pt,
      table.hline(stroke: 0.8pt), [Symbol], [Meaning and range], table.hline(stroke: 0.4pt),
      [$q_"data", x_0$], [Data distribution and clean sample $x_0 ~ q_"data"$ in $[-1,1]^D$.],
      [$q, p_theta, theta$], [Fixed forward distribution, learned generative distribution, and trainable real-valued network parameters.],
      [$t, T$], [Discrete step $t in {1,...,T}$; $T = 1000$ in our experiments.],
      [$beta_t, alpha_t$], [Forward variance $0 < beta_t < 1$; $alpha_t = 1 - beta_t$.],
      [$overline(alpha)_t$], [Product $product_(s=1)^t alpha_s$, with $overline(alpha)_0 = 1$.],
      [$bold(I), epsilon$], [$D times D$ identity matrix and independent noise $epsilon ~ cal(N)(bold(0),bold(I))$.],
      [$epsilon_theta, mu_theta, sigma_t^2$], [Predicted noise and reverse mean in $bb(R)^D$, and fixed reverse variance $sigma_t^2 >= 0$.],
      [$bb(E), norm(v)^2$], [Expectation over stated random draws; squared Euclidean norm $sum_(j=1)^D v_j^2$.],
      table.hline(stroke: 0.8pt),
    )
  },
) <tab:symbols>

The forward process adds independent Gaussian increments. Following Eqs. (2) and (4) of @ho2020ddpm,

$ q(x_(1:T) | x_0) = product_(t=1)^T q(x_t | x_(t-1)), quad
  q(x_t | x_(t-1)) = cal(N)(x_t; sqrt(1-beta_t) x_(t-1), beta_t bold(I)). $ <eq:forward>

Here $cal(N)(x;mu,Sigma)$ denotes a Gaussian density evaluated at $x$, with mean $mu$ and covariance $Sigma$. The experiments use the linear schedule $beta_t = 10^(-4) + (t-1)(0.02-10^(-4))/(T-1)$. Thus $sqrt(beta_t)$ is the standard deviation of one noise increment. Small steps motivate a Gaussian approximation for learned reverse transitions; they do not make the exact reverse conditional Gaussian for arbitrary data.

The accumulated forward noise has a closed form:

$ q(x_t | x_0) = cal(N)(x_t; sqrt(overline(alpha)_t) x_0, (1-overline(alpha)_t) bold(I)), $ <eq:closedform>
$ x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon. $ <eq:reparam>

For this finite schedule, $overline(alpha)_(1000) approx 4.0358 times 10^(-5)$ and the signal coefficient is approximately $0.00635$. The terminal state is therefore close to standard Gaussian noise, with a small residual dependence on the image. The forward distribution is fixed rather than learned. @fig:ddpm-chain illustrates both directions.

#figure(
  image("../figures/fig1_ddpm_chain.pdf", width: 100%),
  caption: [Conceptual forward and reverse chains, drawn by the authors based on @ho2020ddpm, Figure 2. The small grids are stylized illustrations, not Gaussian simulations. Reverse sampling generates a new sample rather than recovering a particular original image.],
) <fig:ddpm-chain>

=== Reverse Model and Noise-Prediction Objective

The learned model starts from $p(x_T)=cal(N)(bold(0),bold(I))$ and uses

$ p_theta(x_(0:T)) = p(x_T) product_(t=1)^T p_theta(x_(t-1)|x_t), quad
  p_theta(x_(t-1)|x_t)=cal(N)(x_(t-1);mu_theta(x_t,t),sigma_t^2 bold(I)). $ <eq:reverse>

For $t>1$, the posterior conditioned on the clean image is Gaussian. Its mean and variance, from Eq. (7) of @ho2020ddpm, are

$ tilde(mu)_t(x_t,x_0) = frac(sqrt(overline(alpha)_(t-1)) beta_t,1-overline(alpha)_t)x_0
 + frac(sqrt(alpha_t)(1-overline(alpha)_(t-1)),1-overline(alpha)_t)x_t,
 quad tilde(beta)_t=frac(1-overline(alpha)_(t-1),1-overline(alpha)_t)beta_t. $ <eq:posterior>

We use $sigma_t^2=tilde(beta)_t$ during sampling. Noise prediction provides a convenient mean parameterization (their Eq. (11)):

$ mu_theta(x_t,t)=frac(1,sqrt(alpha_t))(x_t-frac(beta_t,sqrt(1-overline(alpha)_t))epsilon_theta(x_t,t)). $ <eq:mu_param>

This does not automatically give an unweighted loss. For intermediate steps, the negative evidence lower bound (negative ELBO), which upper-bounds negative log-likelihood, contains terms of the form

$ bb(E)[w_t norm(epsilon-epsilon_theta(x_t,t))^2] + "constant", quad
 w_t=frac(beta_t^2,2 sigma_t^2 alpha_t(1-overline(alpha)_t)), quad t>1. $ <eq:weighted>

Ho et al. deliberately discard these timestep-dependent weights to obtain their simplified objective (Eqs. (12) and (14) of @ho2020ddpm):

$ L_"simple"=bb(E)_(t,x_0,epsilon)[norm(epsilon-epsilon_theta(x_t,t))^2]. $ <eq:lsimple>

Training samples $x_0 ~ q_"data"$, $t$ uniformly from ${1,...,T}$, and independent standard Gaussian $epsilon$. The network receives $x_t$ from @eq:reparam and $t$; an optimizer updates $theta$ to reduce the noise error. Our conditional models also receive label $y$. The code averages squared error over coordinates and the mini-batch; this rescales the sum in @eq:lsimple by a constant for each fixed representation size.

Why can this generate new images? Under squared error, the optimal predictor estimates the conditional mean of the injected noise given the noisy input. Across images and noise levels, it learns statistical structure that guides reverse transitions toward likely data. It is not given the original image during generation and does not need to identify the exact noise realization that produced an arbitrary input. This is the noise-prediction connection to denoising score matching discussed by @ho2020ddpm.

=== Training and Sampling Steps

The following procedures summarize Algorithms 1-2 of @ho2020ddpm.

1. Training: draw a mini-batch of clean images and labels, random timesteps, and independent noise.
2. Form each noisy input with @eq:reparam; predict its noise using the image, timestep, and label.
3. Average squared prediction errors, backpropagate, and update the network. Repeat over training batches.
4. Sampling: draw $x_T ~ cal(N)(bold(0),bold(I))$, choose the label if conditional, and iterate $t=T,...,1$ using

$ x_(t-1)=mu_theta(x_t,t)+sigma_t xi_t,
 quad xi_t ~ cal(N)(bold(0),bold(I)) " for " t>1, quad xi_1=bold(0). $ <eq:sampling>

Each reverse-step noise draw is independent. The implementation first estimates $hat(x)_0=(x_t-sqrt(1-overline(alpha)_t)epsilon_theta)/sqrt(overline(alpha)_t)$, clips it to $[-1,1]$, and uses this estimate in @eq:posterior. At $t=1$ it returns the clean estimate without extra noise. Thus the implemented mean includes clipping, unlike the unclipped algebraic form of @eq:mu_param. Code index $k$ denotes mathematical step $t=k+1$.

== Step-by-Step Worked Numerical Example

We use the first MNIST test image labeled 3 @mnist, with zero-based pixel coordinate $(10,14)$, following the saved output of `step0_worked_example.py`. Its raw intensity is 236. A CPU PyTorch noise draw with seed 42 supplies the selected coordinate $epsilon=-0.714962$. This is one realization, not an expected noise value. The one-dimensional forward step is calculated as follows:

1. Normalize: $x_0=236/127.5-1=0.850980$.
2. Choose $beta_1=0.0001$, so $sqrt(1-beta_1) approx 0.999950$ and $sqrt(beta_1)=0.010000$.
3. Calculate the conditional mean: $mu_1=0.999950 times 0.850980=0.850938$.
4. Calculate the noise increment: $0.010000 times (-0.714962)=-0.007150$.
5. Add the terms:

$ x_1=0.850938-0.007150=0.843788. $ <eq:example>

The conditional expectation is $bb(E)[x_1|x_0]=0.850938$, because $bb(E)[epsilon]=0$; its conditional variance is $0.0001$. Repeating the experiment would give different samples around this mean. More generally, for independent noise, $"Var"(x_t)=overline(alpha)_t "Var"(x_0)+(1-overline(alpha)_t)$ for each coordinate. It equals one only if the original coordinate has unit variance; normalization alone does not imply that assumption.

At $t=50$, the recorded $overline(alpha)_(50)=0.971016$ gives
$ x_(50)=0.985401 times 0.850980+0.170248 times (-0.714962)=0.716836. $
Reusing the same $epsilon$ couples two valid marginal samples for illustration; it is not a sequential Markov path.

The saved untrained U-Net output at that pixel, $[epsilon_theta(x_1,1,y=3)]_(10,14)=0.421943$, gives a residual of $-1.136905$ and squared error $1.292554$. Here $x_1$ in the predictor denotes the whole noisy image. The recorded image-level mean is

$ ell_(t=1)=frac(1,784)sum_(r=0)^27 sum_(c=0)^27
 (epsilon_(r,c)-[epsilon_theta(x_1,1,3)]_(r,c))^2=1.134913. $

This is a single-image, single-timestep estimate of the coordinate-averaged objective, not the full training expectation. Numerical values are rounded to six decimal places; the saved computation uses unrounded tensors.

== LDM: Two-Stage Latent Diffusion <sec:ldm>

Rombach et al. @rombach2022ldm (Section 3) first train an autoencoder, then train diffusion in its frozen latent space. For an image $x in bb(R)^(H times W times C)$, the encoder $cal(E)$ produces a spatial latent in $bb(R)^(h times w times c)$, where $h,w,c$ are latent height, width, and channels. The downsampling factor is $f=H/h=W/w$. Reconstruction and generation are distinct:

$ z_0=cal(E)(x), quad tilde(x)=cal(D)(cal(E)(x)), quad
 x_"gen"=cal(D)(z_0^"gen"). $ <eq:vae-dec>

The generated latent $z_0^"gen"$ comes from reverse diffusion, not from encoding a particular input image. Reducing each spatial dimension by $f$ gives $f^2$ fewer spatial positions, while the scalar compression ratio also depends on channel count. For our $1 times 28 times 28$ images and $4 times 7 times 7$ latents, these reductions are $16 times$ and $4 times$, respectively.

The paper uses perceptual and adversarial reconstruction terms with either mild KL regularization or vector quantization. Our smaller VAE uses only pixel MSE and KL regularization, so it is an illustrative simplification. Its encoder outputs $mu$ and $log sigma^2$ and samples $z=mu+sigma dot.o eta$, with independent $eta ~ cal(N)(bold(0),bold(I))$; $dot.o$ denotes coordinatewise multiplication. The implemented objective is

$ L_"VAE"=frac(1,D)norm(x-tilde(x))^2
 -frac(beta_"KL",2D_z)sum_(j=1)^(D_z)(1+log sigma_j^2-mu_j^2-sigma_j^2), $ <eq:lvae>

where $D_z=h times w times c$, $beta_"KL"=5 times 10^(-4)$, and both terms are also averaged over the batch. A mild KL penalty regularizes the latent distribution without guaranteeing exact standard normality @kingma2014vae. For diffusion, the code caches deterministic encoder means and uses $z_0=s mu(x)$, with recorded global scale $s=1.160420$. Generated latents are divided by $s$ before decoding. The frozen VAE is not optimized with the denoiser.

Forward noising follows @eq:reparam with $x$ replaced by $z$. The conditional latent objective is Eq. (3) of @rombach2022ldm:

$ L_"LDM"=bb(E)_(x,y,t,epsilon)[norm(epsilon-epsilon_theta(z_t,t,tau_theta(y)))^2]. $ <eq:lldm>

Here $(x,y)$ is a data-label pair, $t$ is uniform, $epsilon$ is independent standard Gaussian noise, and $tau_theta$ encodes the condition. A smaller latent usually reduces denoiser arithmetic, but total cost also includes autoencoder training, encoding, and decoding. Compression and runtime depend on the setting.

#figure(
  image("../figures/fig2_ldm_pipeline.pdf", width: 100%),
  caption: [Two-stage pipeline drawn by the authors based on @rombach2022ldm, Figure 3. Training uses encoded data; generation begins from random latent noise and ends with one decoder pass. The diagram shows the KL-regularized variant.],
) <fig:ldm-pipeline>

For token conditioning, let $F_i=phi_i(z_t) in bb(R)^(N times d_i)$ be flattened U-Net features at layer $i$, with $N=h_i w_i$ spatial positions. The encoder output $U=tau_theta(y) in bb(R)^(M times d_tau)$ contains $M$ tokens. For one attention head of width $d$, the row-oriented projections are

$ Q=F_i W_Q, quad K=U W_K, quad V=U W_V,
 quad "Attention"(Q,K,V)="softmax"(frac(Q K^T,sqrt(d)))V. $ <eq:crossattn>

The learned matrices have shapes $W_Q in bb(R)^(d_i times d)$ and $W_K,W_V in bb(R)^(d_tau times d)$. Thus $Q in bb(R)^(N times d)$, $K,V in bb(R)^(M times d)$, and softmax normalizes each row over the $M$ conditioning tokens. Multi-head attention repeats this operation for separate heads. The original text-conditioned LDM uses a BERT tokenizer and transformer encoder; spatial conditions such as masks or low-resolution images can instead enter through concatenation @rombach2022ldm.

Our two U-Nets both add the class embedding to the time embedding and apply four-head cross-attention after the first encoder block and at the bottleneck. Each class is represented by only one token ($M=1$). Softmax over one key is always one, so the attention output is a class-dependent value broadcast across positions. This can supply class information, but does not demonstrate the token-selection or spatial alignment possible with a multi-token text encoder.
