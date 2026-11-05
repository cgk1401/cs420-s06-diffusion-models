#import "../helpers.typ": guideline, fig-placeholder

#block(width: 100%)[
  #set par(first-line-indent: 0pt)
  #align(center)[
    #text(size: 9pt, weight: "bold", tracking: 0.5pt, fill: luma(90))[
      CS420: INTRODUCTION TO ARTIFICIAL INTELLIGENCE — SEMINAR RESEARCH REPORT [S06]
    ]
    #v(7pt)
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
        *Author Name 1* \
        #text(size: 9pt, fill: luma(80))[Student ID: XXXXXXXX]
      ],
      [
        *Author Name 2* \
        #text(size: 9pt, fill: luma(80))[Student ID: XXXXXXXX]
      ],
      [
        *Thi Thanh Huyen Phan* \
        #text(size: 9pt, fill: luma(80))[Student ID: 24125031]
      ],
    )
    #v(6pt)
    #text(size: 9.2pt)[
      Faculty of Information Technology, University of Science, VNU-HCM, Ho Chi Minh City, Vietnam \
      `{author1, author2, author3}@student.hcmus.edu.vn`
    ]
  ]

  #v(14pt)

  // Academic Abstract Block
  #pad(x: 1.0cm)[
    #align(center)[#text(size: 10.2pt, weight: "bold")[Abstract]]
    #v(4pt)
    #set text(size: 9.5pt)
    #guideline("Abstract | ~150–200 words")[
      - *Context:* Briefly introduce generative AI in image synthesis and the emergence of diffusion models.
      - *Core Question & Scope:* State how iưterative Gaussian denoising enables novel image generation in DDPM @ho2020ddpm, and how Latent Diffusion Models (LDM) @rombach2022ldm reduce computational demands.
      - *Experiments:* Mention the 1D worked numerical example, 5-level forward noising / reverse denoising demonstrations, and pixel-vs.-latent comparison.
      - *Main Takeaway:* Summarize key findings regarding sampling speed, quality, prompt control, and limitations.
    ]
    #v(4pt)
    #text(size: 9.2pt)[
      *Index Terms*—Generative AI, Denoising Diffusion Probabilistic Models (DDPM), Latent Diffusion Models (LDM), Variational Autoencoder, Cross-Attention.
    ]
  ]

  #v(10pt)
  #line(length: 100%, stroke: 0.5pt + luma(180))
]
