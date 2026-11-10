// ============================================================================
// FIGURE 2: TWO-STAGE LATENT DIFFUSION MODEL (LDM) PIPELINE
// Formal Grayscale Academic Style — Balanced Full-Width Layout
// ============================================================================

#set page(width: 18cm, height: auto, margin: (x: 0.4cm, y: 0.5cm))
#set text(
  font: ("New Computer Modern", "Linux Libertine", "Times New Roman"),
  size: 8pt,
  lang: "en",
)

// Academic Grayscale Palette
#let c-black   = luma(0%)
#let c-dark    = luma(25%)
#let c-med     = luma(50%)
#let c-light   = luma(82%)
#let c-lighter = luma(94%)
#let c-bg      = luma(98%)

// Helper: VAE-style Trapezoid Block (Encoder / Decoder)
#let trapezoid(w: 42pt, h-left: 46pt, h-right: 24pt, fill-c: white, stroke-c: c-black, label: [], sublabel: []) = {
  let max-h = calc.max(h-left, h-right)
  let y-tl = (max-h - h-left) / 2
  let y-bl = y-tl + h-left
  let y-tr = (max-h - h-right) / 2
  let y-br = y-tr + h-right

  box(width: w, height: max-h)[
    #place(top + left)[
      #polygon(
        fill: fill-c,
        stroke: 0.75pt + stroke-c,
        (0pt, y-tl),
        (w, y-tr),
        (w, y-br),
        (0pt, y-bl),
      )
    ]
    #place(center + horizon)[
      #align(center)[
        #text(weight: "bold", size: 7.2pt, fill: stroke-c)[#label] \
        #if sublabel != [] {
          v(-2.5pt)
          text(size: 6.2pt, fill: stroke-c)[#sublabel]
        }
      ]
    ]
  ]
}

// Helper: VAE Gaussian Parameters & Reparameterization Module
#let vae-bottleneck() = {
  stack(dir: ttb, spacing: 2pt,
    grid(
      columns: (18pt, 18pt),
      gutter: 2.5pt,
      rect(width: 18pt, height: 13pt, stroke: 0.6pt + c-dark, fill: c-lighter, radius: 1.5pt)[
        #align(center + horizon)[#text(size: 6.5pt, weight: "bold")[$mu$]]
      ],
      rect(width: 18pt, height: 13pt, stroke: 0.6pt + c-dark, fill: c-lighter, radius: 1.5pt)[
        #align(center + horizon)[#text(size: 6.5pt, weight: "bold")[$sigma$]]
      ],
    ),
    text(size: 6.5pt, fill: c-dark)[$arrow.b$],
    box(
      fill: white,
      stroke: 0.7pt + c-black,
      radius: 2pt,
      inset: (x: 4pt, y: 2.5pt),
    )[
      #align(center)[
        #text(size: 5.8pt, weight: "bold")[$z = mu + sigma ⊙ epsilon$] \
        #text(size: 5pt, fill: c-med)[$epsilon tilde cal(N)(0, bold(I))$]
      ]
    ],
    text(size: 5.2pt, fill: c-med)[Reparameterization]
  )
}

// Helper: Multi-slice Tensor Representation (Latent Space z)
#let latent-tensor(label: $z$) = {
  box(width: 30pt, height: 34pt)[
    #for i in (2, 1, 0) {
      let offset = i * 3.2pt
      place(top + left, dx: offset, dy: (2 - i) * 3.2pt)[
        #rect(
          width: 22pt,
          height: 25pt,
          radius: 1.5pt,
          fill: luma(100% - i * 5%),
          stroke: 0.65pt + c-black,
        )[
          #if i == 0 {
            align(center + horizon)[
              #text(weight: "bold", size: 7.5pt, fill: c-black)[#label]
            ]
          }
        ]
      ]
    }
  ]
}

// Helper: Input/Output Images
#let image-box(symbol: $x$, caption: "Input Image") = {
  stack(dir: ttb, spacing: 3pt,
    box(
      width: 36pt,
      height: 36pt,
      radius: 2pt,
      stroke: 0.75pt + c-black,
      fill: gradient.linear(luma(96%), luma(75%), luma(88%), angle: 45deg),
    )[
      #align(center + horizon)[
        #text(weight: "bold", size: 8.5pt, fill: c-dark)[#symbol]
      ]
    ],
    text(size: 6.2pt, weight: "medium", fill: c-dark)[#caption],
  )
}

// Helper: Noisy Latent Tensor
#let noisy-latent-box(label: $z_T$, caption: "Noise") = {
  stack(dir: ttb, spacing: 3pt,
    box(
      width: 28pt,
      height: 28pt,
      radius: 2pt,
      stroke: 0.75pt + c-dark,
      fill: c-lighter,
    )[
      #place(center + horizon)[
        #grid(
          columns: (3.8pt, 3.8pt, 3.8pt, 3.8pt),
          rows: (3.8pt, 3.8pt, 3.8pt, 3.8pt),
          gutter: 1pt,
          ..range(16).map(i => rect(
            fill: luma(calc.rem(i * 47 + 19, 85) * 1%),
            radius: 0.5pt,
            stroke: none
          ))
        )
      ]
      #place(center + horizon)[
        #rect(fill: rgb(255, 255, 255, 200), radius: 1.5pt, inset: (x: 2pt, y: 1pt))[
          #text(weight: "bold", size: 6.8pt, fill: c-black)[#label]
        ]
      ]
    ],
    text(size: 6.2pt, fill: c-med)[#caption]
  )
}

// Helper: Denoising U-Net Architecture Block
#let unet-block() = {
  box(
    stroke: 0.8pt + c-black,
    fill: white,
    radius: 3pt,
    inset: (x: 8pt, y: 5pt),
  )[
    #stack(dir: ttb, spacing: 2.8pt,
      grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        text(weight: "bold", size: 7.2pt, fill: c-black)[Denoising U-Net],
        rect(stroke: 0.5pt + c-dark, fill: c-lighter, radius: 1.5pt, inset: (x: 3pt, y: 1.5pt))[
          #text(size: 5.8pt, fill: c-dark)[Time step $t$]
        ]
      ),
      v(1pt),
      grid(
        columns: (14pt, 24pt, 14pt),
        align: horizon + center,
        // Downsampling
        stack(dir: ttb, spacing: 2pt,
          rect(width: 14pt, height: 10pt, fill: c-lighter, stroke: 0.5pt + c-dark, radius: 1pt)[#align(center+horizon)[#text(size: 4.8pt)[Down]]],
          text(size: 5.2pt, fill: c-med)[$arrow.b$],
          rect(width: 12pt, height: 10pt, fill: c-light, stroke: 0.5pt + c-dark, radius: 1pt)[#align(center+horizon)[#text(size: 4.8pt)[Down]]],
        ),
        // Skip Connections & Cross-Attention
        stack(dir: ttb, spacing: 2pt,
          line(length: 22pt, stroke: (paint: c-med, thickness: 0.5pt, dash: "densely-dotted")),
          text(size: 4.8pt, fill: c-med)[skip],
          line(length: 22pt, stroke: (paint: c-med, thickness: 0.5pt, dash: "densely-dotted")),
          box(
            fill: c-lighter,
            stroke: 0.65pt + c-black,
            radius: 2pt,
            inset: (x: 2.5pt, y: 2.5pt),
          )[
            #text(size: 5.2pt, weight: "bold", fill: c-black)[Cross-Attn]
          ]
        ),
        // Upsampling
        stack(dir: ttb, spacing: 2pt,
          rect(width: 14pt, height: 10pt, fill: c-lighter, stroke: 0.5pt + c-dark, radius: 1pt)[#align(center+horizon)[#text(size: 4.8pt)[Up]]],
          text(size: 5.2pt, fill: c-med)[$arrow.t$],
          rect(width: 12pt, height: 10pt, fill: c-light, stroke: 0.5pt + c-dark, radius: 1pt)[#align(center+horizon)[#text(size: 4.8pt)[Up]]],
        ),
      ),
      v(1pt),
      text(size: 5.8pt, fill: c-dark)[$epsilon_theta(z_t, t, tau_theta(y))$]
    )
  ]
}

// Helper: Directional Arrow
#let flow-arrow(lbl: [], sub: []) = {
  stack(dir: ttb, spacing: 1pt,
    if lbl != [] { text(size: 6pt, weight: "medium", fill: c-dark)[#lbl] },
    text(size: 8.5pt, fill: c-dark)[$arrow.r$],
    if sub != [] { text(size: 5.2pt, fill: c-med)[#sub] },
  )
}

// Helper: Formal Section Header
#let section-header(title, subtitle) = {
  rect(
    width: 100%,
    fill: c-lighter,
    stroke: (left: 2.5pt + c-black, rest: 0.5pt + c-light),
    radius: (right: 2pt),
    inset: (x: 6pt, y: 3.5pt),
  )[
    #text(weight: "bold", size: 7.5pt, fill: c-black)[#title]
    #h(6pt)
    #text(size: 6.5pt, fill: c-med)[#subtitle]
  ]
}

// ============================================================================
// MAIN PIPELINE DIAGRAM
// ============================================================================
#let fig-ldm-pipeline() = {
  block(
    width: 100%,
    fill: white,
    stroke: 0.7pt + c-light,
    radius: 4pt,
    inset: (x: 10pt, y: 8pt),
  )[
    #set par(first-line-indent: 0pt, leading: 0.38em)

    // ------------------------------------------------------------------------
    // STAGE 1: VAE PERCEPTUAL COMPRESSION
    // ------------------------------------------------------------------------
    #section-header(
      [STAGE 1: PERCEPTUAL COMPRESSION (VARIATIONAL AUTOENCODER)],
      [KL-regularized latent space via probabilistic encoding and reparameterization]
    )

    #v(6pt)

    #grid(
      columns: (1fr, auto, 1.1fr, auto, 1.25fr, auto, 1fr, auto, 1.1fr, auto, 1fr),
      align: center + horizon,
      gutter: 2pt,
      // 1. Input Image
      image-box(symbol: $x$, caption: [$x in bb(R)^(H times W times 3)$]),
      flow-arrow(lbl: [Encode], sub: [$f = H/h$]),
      
      // 2. Encoder E
      trapezoid(
        w: 42pt, h-left: 42pt, h-right: 22pt,
        fill-c: c-lighter, stroke-c: c-black,
        label: [Encoder], sublabel: [$cal(E)$]
      ),
      flow-arrow(),

      // 3. VAE Gaussian Parameters & Reparameterization Module
      vae-bottleneck(),
      flow-arrow(),

      // 4. Sampled Latent Tensor z
      stack(dir: ttb, spacing: 2pt,
        latent-tensor(label: $z$),
        text(size: 6.2pt, weight: "bold", fill: c-black)[Latent $z$],
        text(size: 5.5pt, fill: c-med)[$z in bb(R)^(h times w times c)$],
      ),
      flow-arrow(),

      // 5. Decoder D
      trapezoid(
        w: 42pt, h-left: 22pt, h-right: 42pt,
        fill-c: c-lighter, stroke-c: c-black,
        label: [Decoder], sublabel: [$cal(D)$]
      ),
      flow-arrow(lbl: [Reconstruct]),

      // 6. Output Reconstructed Image
      image-box(symbol: $tilde(x)$, caption: [$tilde(x) = cal(D)(z)$]),
    )

    #v(6pt)
    #line(length: 100%, stroke: 0.5pt + c-light)
    #v(6pt)

    // ------------------------------------------------------------------------
    // STAGE 2: LATENT DIFFUSION (ALIGNED TO STAGE 1)
    // ------------------------------------------------------------------------
    #section-header(
      [STAGE 2: LATENT DIFFUSION PROCESS],
      [Iterative denoising directly in latent space, conditioned via Cross-Attention]
    )

    #v(6pt)

    #grid(
      columns: (1fr, auto, 1fr, auto, 2fr, auto, 1fr),
      align: center + horizon,
      gutter: 4pt,
      // 1. Prior Noise
      noisy-latent-box(label: $z_T$, caption: [$z_T tilde cal(N)(0, bold(I))$]),
      
      // 2. Denoising sequence
      flow-arrow(lbl: [Denoise], sub: [$t = T -> 0$]),
      
      // 3. Noisy intermediate state
      noisy-latent-box(label: $z_t$, caption: [Latent $z_t$]),
      
      // 4. Input into U-Net
      flow-arrow(),
      
      // 5. CENTER COLUMN: U-NET + DIRECT INTEGRATED CONDITIONING
      stack(dir: ttb, spacing: 3.5pt,
        unet-block(),
        text(size: 7.5pt, fill: c-dark)[$arrow.t$],
        box(
          stroke: 0.6pt + c-dark,
          fill: c-lighter,
          radius: 2pt,
          inset: (x: 6pt, y: 3.5pt),
        )[
          #grid(
            columns: (auto, auto, auto),
            align: center + horizon,
            gutter: 5pt,
            text(size: 5.8pt, weight: "bold", fill: c-black)[Condition $y$\ #text(weight: "regular", size: 5pt)[(Text / Mask)]],
            text(size: 7pt, fill: c-dark)[$arrow.r$],
            box(
              fill: white,
              stroke: 0.5pt + c-dark,
              radius: 1.5pt,
              inset: (x: 4pt, y: 2pt),
            )[
              #text(size: 5.8pt, weight: "bold", fill: c-black)[Encoder $tau_theta (y)$] \
              #text(size: 4.8pt, fill: c-med)[(e.g., CLIP, T5)]
            ]
          )
        ]
      ),
      
      // 6. Converged step
      flow-arrow(lbl: [Final Step]),
      
      // 7. Output clean latent
      stack(dir: ttb, spacing: 2pt,
        latent-tensor(label: $z_0$),
        text(size: 6.2pt, weight: "bold", fill: c-black)[Denoised $z_0$],
        rect(
          fill: c-lighter,
          stroke: 0.5pt + c-med,
          radius: 1.5pt,
          inset: (x: 3pt, y: 2pt)
        )[
          #text(size: 5.2pt, fill: c-dark)[Forward to $cal(D) arrow.t$]
        ]
      )
    )
  ]
}

#fig-ldm-pipeline()