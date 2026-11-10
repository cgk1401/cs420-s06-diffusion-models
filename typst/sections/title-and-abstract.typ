
#block(width: 100%)[
  #set par(first-line-indent: 0pt)
  #align(center)[
    #text(size: 17pt, weight: "bold")[
      Diffusion Models: From Adding Noise to Generating Images
    ]
    #v(4pt)
    #text(size: 11.2pt, style: "italic", fill: luma(60))[
      From Pixel-Space Denoising Diffusion to High-Resolution Latent Diffusion
    ]
    #v(14pt)

    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 12pt,
      [
        Thanh Huu Nguyen \
        #text(size: 9pt, fill: luma(80))[Student ID: 24125030]
      ],
      [
        Manh Cuong Do \
        #text(size: 9pt, fill: luma(80))[Student ID: 24125054]
      ],
      [
        Thi Thanh Huyen Phan \
        #text(size: 9pt, fill: luma(80))[Student ID: 24125031]
      ],
    )
    #v(6pt)
    #text(size: 9.2pt)[
      Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam \
    ]
  ]

  #v(14pt)

  // Academic Abstract Block
  #pad(x: 1.0cm)[
    #align(center)[#text(size: 10.2pt, weight: "bold")[Abstract]]
    #v(4pt)
    #set text(size: 9.5pt)
    Diffusion models generate images by learning how to remove noise from data. This report studies how this process works and why performing diffusion in a compressed latent space can reduce its computational cost. We review Denoising Diffusion Probabilistic Models (DDPM) and Latent Diffusion Models (LDM), explain their main equations, and provide a step-by-step numerical example. We also examine small class-conditioned implementations on MNIST, using saved experiment records and generated images. In this setting, latent diffusion requires about 14.19 times fewer counted convolution and linear operations per denoising step. However, its recorded sampling time is 1.715 seconds, compared with 1.592 seconds for pixel diffusion, and it requires an additional autoencoder stage. These results illustrate that lower arithmetic cost does not necessarily lead to lower latency at small resolutions. The examples demonstrate basic generation and class conditioning, but do not establish distributional quality or general text-prompt control. We discuss these limits alongside the effects of compression, training data, copyright, and bias.
    #v(4pt)
    #text(size: 9.2pt)[
      Keywords: Generative AI, Denoising Diffusion Probabilistic Models (DDPM), Latent Diffusion Models (LDM), Variational Autoencoder, Cross-Attention.
    ]
  ]

]
