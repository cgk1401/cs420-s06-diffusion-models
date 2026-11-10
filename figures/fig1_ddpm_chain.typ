// ============================================================================
// STANDALONE & IMPORTABLE TYPST DIAGRAM — FIGURE 1: DDPM BIDIRECTIONAL CHAIN
// Conceptual overview only: the mathematical definitions live in Section 3.1.
// File: figures/fig1_ddpm_chain.typ
// Compile to PDF/PNG:
//   typst compile figures/fig1_ddpm_chain.typ figures/fig1_ddpm_chain.pdf
//   typst compile --format png --ppi 300 figures/fig1_ddpm_chain.typ "out-{p}.png"
// ============================================================================

#set page(width: 17.2cm, height: auto, margin: 0.35cm)
#set text(
  font: "New Computer Modern",
  size: 9pt,
  lang: "en",
)

// Helper: Render a 7x7 mini pixel-grid of digit "3" at a given noise level
#let mini-digit-grid(noise-level) = {
  let clean = (
    0.08, 0.88, 0.95, 0.95, 0.95, 0.85, 0.08,
    0.08, 0.12, 0.08, 0.08, 0.12, 0.92, 0.08,
    0.08, 0.08, 0.08, 0.10, 0.20, 0.90, 0.08,
    0.08, 0.10, 0.85, 0.95, 0.95, 0.85, 0.08,
    0.08, 0.08, 0.08, 0.10, 0.18, 0.92, 0.08,
    0.08, 0.15, 0.08, 0.08, 0.12, 0.90, 0.08,
    0.08, 0.88, 0.95, 0.95, 0.95, 0.85, 0.08,
  )

  let noise = (
    0.72, 0.18, 0.85, 0.31, 0.64, 0.12, 0.91,
    0.24, 0.89, 0.42, 0.77, 0.15, 0.58, 0.33,
    0.93, 0.36, 0.68, 0.14, 0.82, 0.47, 0.79,
    0.19, 0.74, 0.29, 0.95, 0.38, 0.66, 0.11,
    0.61, 0.44, 0.87, 0.22, 0.71, 0.94, 0.27,
    0.83, 0.16, 0.53, 0.69, 0.35, 0.21, 0.86,
    0.39, 0.92, 0.17, 0.56, 0.81, 0.49, 0.63,
  )

  let cells = ()
  for i in range(49) {
    let val = (1.0 - noise-level) * clean.at(i) + noise-level * noise.at(i)
    let clamped = calc.min(1.0, calc.max(0.0, val))
    let gray-pct = int(clamped * 100) * 1%
    cells.push(
      rect(
        width: 4.3pt,
        height: 4.3pt,
        fill: luma(gray-pct),
        stroke: none,
      )
    )
  }

  box(
    stroke: 0.6pt + luma(115),
    radius: 1.5pt,
    clip: true,
    grid(
      columns: (4.3pt,) * 7,
      rows: (4.3pt,) * 7,
      ..cells
    )
  )
}

// Helper: Horizontal arrow with direction ("right" or "left") and tone
#let h-arrow(dir: "right", color: luma(8%), dashed: false) = {
  let stroke-style = if dashed {
    (paint: color, thickness: 1pt, dash: "dashed")
  } else {
    (paint: color, thickness: 1.15pt)
  }
  box(width: 100%, height: 8pt)[
    #place(horizon,
      line(
        start: if dir == "right" { (0%, 0pt) } else { (6pt, 0pt) },
        end: if dir == "right" { (100% - 6pt, 0pt) } else { (100%, 0pt) },
        stroke: stroke-style,
      )
    )
    #if dir == "right" {
      place(right + horizon,
        polygon(
          fill: color,
          stroke: none,
          (0pt, -3.2pt),
          (6.5pt, 0pt),
          (0pt, 3.2pt),
        )
      )
    } else {
      place(left + horizon,
        polygon(
          fill: color,
          stroke: none,
          (6.5pt, -3.2pt),
          (0pt, 0pt),
          (6.5pt, 3.2pt),
        )
      )
    }
  ]
}

// Helper: State node card (x_0, x_1, x_{t-1}, x_t, x_T)
#let state-card(math-label, sub-label, noise-lvl, border-color) = {
  rect(
    width: 100%,
    fill: luma(100%),
    stroke: 1.15pt + border-color,
    radius: 4pt,
    inset: (x: 4pt, y: 6pt),
  )[
    #set par(first-line-indent: 0pt, leading: 0.42em)
    #align(center)[
      #text(size: 10.5pt, weight: "bold")[#math-label]
      #v(4pt)
      #mini-digit-grid(noise-lvl)
      #v(4.5pt)
      #text(size: 7.4pt, fill: luma(65))[#sub-label]
    ]
  ]
}

// Main exported diagram function
#let fig-ddpm-chain() = {
  // Monochrome palette: all tones are pure grayscale (no hue).
  // Forward chain = near-black, reverse chain = dark gray, U-Net = mid-dark gray.
  let fwd = luma(8%)
  let fwd-bg = luma(95%)
  let rev = luma(32%)
  let rev-bg = luma(90%)
  let unet = luma(38%)
  let unet-bg = luma(86%)

  block(
    width: 100%,
    fill: luma(97%),
    stroke: 0.6pt + luma(70%),
    radius: 5pt,
    inset: (x: 10pt, y: 9pt),
  )[
    #set par(first-line-indent: 0pt, leading: 0.45em)

    // ------------------------------------------------------------------------
    // 1. TOP BANNER: FORWARD NOISING PROCESS
    // ------------------------------------------------------------------------
    #rect(
      width: 100%,
      fill: fwd-bg,
      stroke: (left: 2.8pt + fwd, rest: 0.5pt + luma(72%)),
      radius: 3pt,
      inset: (x: 8pt, y: 5pt),
    )[
      #text(weight: "bold", size: 8.4pt, fill: fwd)[
        FORWARD NOISING PROCESS
      ] \
      #text(size: 7.7pt, fill: luma(45))[
        The data is progressively corrupted by Gaussian noise until it becomes indistinguishable from pure noise.
      ]
    ]

    #v(9pt)

    // ------------------------------------------------------------------------
    // 2. MIDDLE CHAIN: STATE NODES & BIDIRECTIONAL ARROWS
    // ------------------------------------------------------------------------
    #grid(
      columns: (1.00fr, 0.95fr, 1.00fr, 0.85fr, 1.00fr, 1.30fr, 1.00fr, 0.85fr, 1.00fr),
      align: center + horizon,
      gutter: 2pt,

      // Node x_0
      state-card([$x_0$], [Clean Data], 0.0, fwd),

      // Arrow x_0 <-> x_1
      [
        #text(size: 7.4pt, fill: fwd)[add noise]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd)
        #v(8pt)
        #h-arrow(dir: "left", color: rev)
        #v(2pt)
        #text(size: 6.6pt, fill: rev)[remove noise]
      ],

      // Node x_1
      state-card([$x_1$], [Slight Noise], 0.18, luma(130)),

      // Dots 1 <-> t-1
      [
        #text(size: 7pt, fill: fwd)[add noise]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd, dashed: true)
        #v(3pt)
        #text(size: 10pt, weight: "bold", fill: luma(90))[$dots.c$]
        #v(3pt)
        #h-arrow(dir: "left", color: rev, dashed: true)
        #v(2pt)
        #text(size: 6.6pt, fill: rev)[remove noise]
      ],

      // Node x_{t-1}
      state-card([$x_(t-1)$], [Noisy State], 0.48, luma(110)),

      // Key Highlighted Transition x_{t-1} <-> x_t
      [
        #box(
          fill: fwd-bg,
          stroke: 0.45pt + fwd,
          radius: 2pt,
          inset: (x: 3.5pt, y: 2pt),
        )[
          #text(size: 7.5pt, weight: "bold", fill: fwd)[add noise]
        ]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd)
        #v(10pt)
        #h-arrow(dir: "left", color: rev)
        #v(2pt)
        #box(
          fill: rev-bg,
          stroke: 0.45pt + rev,
          radius: 2pt,
          inset: (x: 3.5pt, y: 2pt),
        )[
          #text(size: 7.5pt, weight: "bold", fill: rev)[remove noise]
        ]
      ],

      // Node x_t
      state-card([$x_t$], [Noisy State], 0.68, rev),

      // Dots t <-> T
      [
        #text(size: 7pt, fill: fwd)[add noise]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd, dashed: true)
        #v(3pt)
        #text(size: 10pt, weight: "bold", fill: luma(90))[$dots.c$]
        #v(3pt)
        #h-arrow(dir: "left", color: rev, dashed: true)
        #v(2pt)
        #text(size: 6.6pt, fill: rev)[remove noise]
      ],

      // Node x_T
      state-card([$x_T$], [Pure Noise], 1.0, rev),
    )

    #v(9pt)

    // ------------------------------------------------------------------------
    // 3. BOTTOM PANEL: REVERSE DENOISING & U-NET NOISE PREDICTOR
    // ------------------------------------------------------------------------
    #rect(
      width: 100%,
      fill: rev-bg,
      stroke: (left: 2.8pt + rev, rest: 0.5pt + luma(80%)),
      radius: 3pt,
      inset: (x: 8pt, y: 5pt),
    )[
      #grid(
        columns: (auto, 1fr),
        gutter: 8pt,
        align: (left + horizon, right + horizon),
        [
          #text(weight: "bold", size: 8.4pt, fill: rev)[
            REVERSE DENOISING PROCESS
          ] \
          #text(size: 7.7pt, fill: luma(45))[
            A U-Net removes the noise one step at a time, turning pure noise back into a sample.
          ]
        ],
        [
          #box(
            fill: unet-bg,
            stroke: 0.65pt + unet,
            radius: 3pt,
            inset: (x: 7pt, y: 4pt),
          )[
            #align(center)[
              #text(size: 7.5pt, weight: "bold", fill: unet)[U-Net predicts the noise] \
              #v(1pt)
              #text(size: 7.6pt)[$epsilon_theta (x_t, t)$]
            ]
          ]
        ]
      )
    ]
  ]
}

// Render when compiled as a standalone file
#fig-ddpm-chain()
