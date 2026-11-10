= Background and Paper Review
== Discriminative and Generative Models

Given an image $x$ and label $y$, a discriminative model predicts $p(y | x)$ or a decision boundary, for example to classify a digit. A generative model instead models a data distribution such as $p(x)$ or $p(x | y)$ and draws new samples from it. Conditional generation can therefore produce an image consistent with a class label or text description. Classification alone does not provide this sampling capability.

== Qualitative Comparison of Generative Models

GANs train a generator against a discriminator, enabling fast generation but introducing a difficult optimization problem and a risk of mode collapse @goodfellow2014gan @salimans2016gan. Autoregressive models factorize a joint distribution into conditional probabilities and generate components sequentially @oord2016pixelrnn. Variational Autoencoders (VAEs) learn a probabilistic encoder and decoder with a latent prior; generation draws from that prior and decodes a sample @kingma2014vae. Diffusion models learn denoising across noise levels and use repeated network evaluations for generation @sohl2015deep @ho2020ddpm. @tab:gen-comparison summarizes typical trade-offs, not a universal ranking or a controlled benchmark.

#figure(
  kind: table,
  caption: [Qualitative synthesis by the authors from @goodfellow2014gan, @salimans2016gan, @oord2016pixelrnn, @kingma2014vae, and @ho2020ddpm. Outcomes depend on architecture, data, and evaluation.],
  {
    set text(size: 9pt)
    set par(justify: false, first-line-indent: 0pt)
    table(
      columns: (0.9fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr), stroke: none, align: left, inset: 5pt,
      table.hline(stroke: 0.8pt),
      [Criterion], [GAN], [Autoregressive], [VAE], [Diffusion],
      table.hline(stroke: 0.4pt),
      [Learning], [Adversarial], [Conditional likelihood], [Variational bound], [Denoising objective],
      [Quality], [Can be sharp], [Can be high], [May be smoother], [Can be high],
      [Training], [Can be unstable], [Direct likelihood objective], [Variational optimization], [No adversarial game],
      [Coverage], [Mode-collapse risk], [Depends on fit], [Depends on prior and fit], [Depends on fit; no guarantee],
      [Sampling], [One generator pass], [Sequential components], [One decoder pass], [Many denoising steps],
      table.hline(stroke: 0.8pt),
    )
  },
) <tab:gen-comparison>

== Review of Core Papers: From DDPM to LDM

Sohl-Dickstein et al. @sohl2015deep introduced the earlier diffusion probabilistic framework. Ho, Jain, and Abbeel @ho2020ddpm develop an effective noise-prediction parameterization and a simplified training objective, connecting diffusion to denoising score matching and Langevin-like sampling. Their unconditional CIFAR-10 result has Inception Score (IS) $9.46 plus.minus 0.11$ and Fréchet Inception Distance (FID) 3.17 (Table 1 of their paper). FID compares feature distributions of real and generated samples; lower values indicate closer agreement under that evaluation. The paper reports competitive $256 times 256$ LSUN samples, but repeated pixel-space denoising remains costly. These published findings are not reproduced here.

Rombach et al. @rombach2022ldm separate perceptual compression from diffusion-based generation. A pretrained autoencoder provides a smaller spatial representation, and the diffusion model learns its distribution. Their cross-attention interface supports token-based conditions such as text and layouts; spatially aligned conditions can also be concatenated. Compression introduces a trade-off: small downsampling factors leave expensive representations, whereas excessive compression loses detail (their Section 4.1 and Figures 5-6). Their Table 6 provides a matched inpainting comparison: on Places at $256 times 256$, LDM-1 and KL-regularized LDM-4 with attention have training throughputs of 0.11 and 0.32 samples/s and sampling throughputs of 0.26 and 0.97 samples/s. The settings have fixed model parameter counts; hardware/batch details are specified in the paper's supplement. These task-specific results support efficiency benefits without defining a universal speedup.

For class-conditioned ImageNet generation, their Table 3 reports FID 10.56 for LDM-4 and 3.60 for LDM-4-G with classifier-free guidance (scale 1.5, 250 sampling steps). These are separate from DDPM's CIFAR-10 result and cannot be compared as a common quality ranking. Together, the papers motivate two distinct design choices: learning a denoising-based generator and choosing an efficient representation in which to run it. Faster samplers such as DDIM can additionally reduce network evaluations in either space @song2021ddim.
