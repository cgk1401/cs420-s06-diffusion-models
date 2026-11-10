
= Background and Paper Review
== Discriminative and Generative Models

Discriminative and generative models differ in their learning objectives. Given an input image $x$ and its label $y$, a discriminative model estimates the conditional distribution $p(y | x)$ and uses it to assign the input to a class, such as distinguishing a cat from a dog. This capability does not imply the ability to synthesize new images. In contrast, a generative model learns the distribution of the data itself - either the unconditional distribution $p(x)$ or a conditional distribution $p(x | y)$ - and can therefore draw new samples from the learned distribution. For example, an unconditional generative model can synthesize novel human faces, whereas a conditional model can generate images consistent with a class label or a text description. In essence, discriminative models estimate the probability of a label given an input, while generative models estimate the distribution from which new inputs can be sampled.

== Qualitative Comparison of Generative Models

There are several types of generative models, and each uses a different approach to learn and generate data. GANs consist of two networks: a generator and a discriminator. The generator tries to create realistic images, while the discriminator attempts to distinguish generated images from real ones. This process can produce high-quality images relatively quickly. However, GAN training can be unstable, and the model may experience mode collapse, where it generates only a limited variety of images @goodfellow2014gan.

Autoregressive models generate data one component at a time by using conditional probabilities. Each new component depends on the components generated before it. This provides a clear probabilistic way to model data, but it also means that generation must happen sequentially. As a result, generating high-dimensional data can take considerable time.

Variational Autoencoders (VAEs) work by learning how to compress data into a latent representation and then reconstruct it using a decoder. Once trained, the decoder can also be used to generate new samples. VAEs usually have stable training and relatively fast sampling. However, their generated images may sometimes look less sharp compared with those produced by other generative models @kingma2014vae.

Diffusion Models take another approach. They learn to generate images by gradually removing noise over multiple steps. Starting from random noise, the model repeatedly predicts and removes noise until an image is formed. This method can produce detailed and diverse images without using the adversarial training process of GANs. The main disadvantage is that image generation is relatively slow because the model needs to perform many denoising steps @sohl2015deep @ho2020ddpm.

Table @tab:gen-comparison summarizes the main differences between these generative approaches. These comparisons describe general characteristics rather than fixed results, since the actual performance of each model can depend on the dataset, architecture, and evaluation conditions.

#figure(
  kind: table,
  caption: [Qualitative comparison of major generative modeling families, synthesized from @goodfellow2014gan, @kingma2014vae, @sohl2015deep, and @ho2020ddpm. The comparison summarizes the characteristics discussed in Section 2.2.],
  {
    set text(size: 8.8pt)
    table(
      columns: (1.25fr, 1.1fr, 1.1fr, 1.1fr, 1.25fr),
      align: (left, left, left, left, left),
      stroke: none,
      inset: (x: 5pt, y: 5.5pt),

      table.hline(stroke: 0.9pt),
      [Criterion], [GANs], [Autoregressive], [VAEs], [Diffusion Models],
      table.hline(stroke: 0.5pt),

      [Core Principle],
      [Adversarial \ #text(size: 7.3pt, fill: luma(90))[Generator against discriminator]],
      [Sequential \ #text(size: 7.3pt, fill: luma(90))[Conditional step-by-step]],
      [Autoencoding \ #text(size: 7.3pt, fill: luma(90))[Latent compression and decoding]],
      [Iterative Denoising \ #text(size: 7.3pt, fill: luma(90))[Multi-step noise removal]],

      [Sample Quality],
      [High \ #text(size: 7.3pt, fill: luma(90))[Sharp realistic images]],
      [High \ #text(size: 7.3pt, fill: luma(90))[High-fidelity components]],
      [Moderate \ #text(size: 7.3pt, fill: luma(90))[May look less sharp]],
      [High \ #text(size: 7.3pt, fill: luma(90))[Detailed and fine outputs]],

      [Training Stability],
      [Unstable \ #text(size: 7.3pt, fill: luma(90))[Adversarial dynamics]],
      [Stable \ #text(size: 7.3pt, fill: luma(90))[Direct probabilistic loss]],
      [Stable \ #text(size: 7.3pt, fill: luma(90))[Consistent optimization]],
      [Stable \ #text(size: 7.3pt, fill: luma(90))[No adversarial training]],

      [Mode Coverage],
      [Limited \ #text(size: 7.3pt, fill: luma(90))[Risk of mode collapse]],
      [High \ #text(size: 7.3pt, fill: luma(90))[Broad data coverage]],
      [High \ #text(size: 7.3pt, fill: luma(90))[Smooth latent prior]],
      [High \ #text(size: 7.3pt, fill: luma(90))[Diverse sample output]],

      [Sampling Speed],
      [Fast \ #text(size: 7.3pt, fill: luma(90))[Single generator pass]],
      [Slow \ #text(size: 7.3pt, fill: luma(90))[Sequential generation]],
      [Fast \ #text(size: 7.3pt, fill: luma(90))[Single decoder pass]],
      [Slow \ #text(size: 7.3pt, fill: luma(90))[Many denoising steps]],

      table.hline(stroke: 0.9pt),
    )
  },
) <tab:gen-comparison>

From this comparison, it can be seen that each generative approach has its own advantages and limitations. GANs and VAEs are generally faster when generating samples, while Diffusion Models often produce high-quality and diverse images but require more computation. This difference becomes especially important when working with high-resolution images. It is also one of the main reasons why LDM was developed, as it aims to reduce the computational cost of diffusion-based image generation.

== Review of Core Papers: From DDPM to LDM

=== DDPM -- Denoising Diffusion Probabilistic Models.

Ho, Jain, and Abbeel @ho2020ddpm introduce a probabilistic approach to image generation based on gradually adding Gaussian noise to images and learning how to reverse this process. One of the main ideas in their paper is to train a neural network to predict the noise added to an image. Instead of directly predicting the original image at every step, the network learns to estimate the noise that needs to be removed. The authors use a simplified training objective, called $L_"simple"$, which minimizes the squared difference between the actual noise and the predicted noise. The paper also explains how diffusion probabilistic models are related to denoising score matching and Langevin dynamics. The mathematical details and training algorithms will be discussed further in Section 3.

The authors evaluate DDPM on several image generation tasks. For unconditional image generation on the CIFAR-10 dataset, they report an Inception Score (IS) of 9.46 and a Fréchet Inception Distance (FID) of 3.17. Their experiments on the LSUN dataset also show that DDPM can achieve competitive results compared with GAN-based methods at the time @ho2020ddpm. These results suggest that learning to reverse a noise process is an effective way to generate images. However, the values are taken from the original paper and are not independently reproduced in this report.

Despite these results, DDPM still has an important limitation. The model performs diffusion directly in pixel space, meaning it processes the original image pixels during denoising. For high-resolution images, this requires a large amount of computation. The generation process is also slow because the neural network must be evaluated repeatedly across many timesteps. While DDPM demonstrates that noise prediction can be used for image generation, its computational cost makes it less practical for some high-resolution applications. This limitation leads to the development of more efficient diffusion models.

=== LDM -- High-Resolution Image Synthesis with Latent Diffusion Models.

To address the computational cost of pixel-space diffusion, Rombach et al. @rombach2022ldm introduce LDM. The main idea is to perform diffusion in a compressed latent space rather than directly on the original image. The authors divide the approach into two main parts: perceptual compression and semantic generation.

First, an autoencoder compresses the input image into a lower-dimensional latent representation. This representation is designed to preserve important visual information while reducing the amount of data that needs to be processed. The diffusion model then learns to add and remove noise in this latent space. After the denoising process is finished, a decoder converts the generated latent representation back into an image.

Because the latent representation is smaller than the original image, the model needs less computation during each denoising step. This makes LDM more efficient than performing diffusion directly in pixel space. Another advantage is that the same autoencoder can be reused for different image generation tasks, which helps separate the image compression process from the generation process.

The LDM paper also introduces cross-attention mechanisms to support conditional image generation. These mechanisms allow the model to use additional information, such as text descriptions or spatial layouts, when generating an image. For example, in text-to-image generation, a user can provide a text prompt, and the model uses that prompt to guide the denoising process. LDM can also be applied to tasks such as inpainting, where missing parts of an image are filled in, and super-resolution, where the goal is to generate a higher-resolution image.

The experiments in the paper show that LDM can achieve competitive results on different image generation tasks while using fewer computational resources than pixel-based diffusion methods @rombach2022ldm. However, LDM also has some limitations. Compressing an image too much may cause the loss of small visual details, which can affect the final image quality. The model still needs multiple denoising steps, so generation is not completely fast. For this reason, both reconstruction quality and sampling efficiency need to be considered when evaluating LDM.

DDPM and LDM are closely related because they both use the idea of learning to generate images through denoising. DDPM shows that a model can learn to predict noise and gradually remove it to generate new images. LDM builds on this idea by moving the diffusion process into a compressed latent space, making it more computationally efficient and allowing additional control through conditioning mechanisms. Studying these two approaches helps explain how diffusion-based image generation works and why there are trade-offs between image quality, computational cost, and flexibility.