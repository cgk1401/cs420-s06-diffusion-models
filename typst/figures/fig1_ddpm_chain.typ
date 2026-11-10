// ============================================================================
// STANDALONE & IMPORTABLE TYPST DIAGRAM — FIGURE 1: DDPM BIDIRECTIONAL CHAIN
// File: figures/fig1_ddpm_chain.typ
// Compile to PDF/PNG:
//   python -c "import typst; typst.compile('figures/fig1_ddpm_chain.typ', output='figures/fig1_ddpm_chain.pdf')"
// ============================================================================

#set page(width: 17.2cm, height: auto, margin: 0.35cm)
#set text(
  font: ("New Computer Modern", "Linux Libertine", "Times New Roman"),
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

// Helper: Horizontal arrow with direction ("right" or "left") and color
#let h-arrow(dir: "right", color: rgb("#1d4ed8"), dashed: false) = {
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
    fill: rgb("#ffffff"),
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
  let fwd-blue = rgb("#1e4e8c")
  let fwd-bg = rgb("#eef4fb")
  let rev-red = rgb("#b43c2f")
  let rev-bg = rgb("#fdf2f0")
  let unet-purple = rgb("#5b3a8c")
  let unet-bg = rgb("#f5f0ff")

  block(
    width: 100%,
    fill: rgb("#fafbfd"),
    stroke: 0.6pt + luma(195),
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
      stroke: (left: 2.8pt + fwd-blue, rest: 0.5pt + rgb("#bfd4ee")),
      radius: 3pt,
      inset: (x: 8pt, y: 5pt),
    )[
      #grid(
        columns: (auto, 1fr),
        gutter: 8pt,
        align: (left + horizon, right + horizon),
        [
          #text(weight: "bold", size: 8.4pt, fill: fwd-blue)[
            FORWARD NOISING PROCESS (Fixed Markov Chain)
          ] \
          #text(size: 7.7pt, fill: luma(45))[
            Step transition ($beta_1, dots, beta_T$): $q(x_t | x_(t-1)) = cal(N)(sqrt(1 - beta_t) x_(t-1), beta_t bold(I))$
          ]
        ],
        [
          #box(
            fill: rgb("#ffffff"),
            stroke: 0.6pt + fwd-blue,
            radius: 3pt,
            inset: (x: 7pt, y: 4pt),
          )[
            #align(center)[
              #text(size: 7.5pt, fill: fwd-blue, weight: "bold")[Reparameterization Form] \
              #v(1pt)
              #text(size: 7.6pt)[
                $x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon, quad epsilon tilde.op cal(N)(bold(0), bold(I))$
              ]
            ]
          ]
        ]
      )
    ]

    #v(6pt)

    // ------------------------------------------------------------------------
    // 2. CLOSED-FORM JUMP BRIDGE FROM x_0 (Col 1) TO x_t (Col 7)
    // ------------------------------------------------------------------------
    // Total weight of columns = 1.05 + 0.75 + 1.05 + 0.70 + 1.05 + 1.35 + 1.05 + 0.70 + 1.05 = 8.75fr
    // Center of Col 1 (x_0) = 0.525fr; Center of Col 7 (x_t) = 6.475fr; Span = 5.95fr; Right remainder = 2.275fr
    #grid(
      columns: (0.525fr, 5.95fr, 2.275fr),
      align: center,
      [],
      [
        #box(width: 100%, height: 18pt)[
          // Horizontal top dashed line
          #place(top + left, dy: 6pt,
            line(length: 100%, stroke: (paint: fwd-blue, thickness: 0.95pt, dash: "dashed"))
          )
          // Left vertical stem going down to x_0
          #place(top + left, dy: 6pt,
            line(start: (0pt, 0pt), end: (0pt, 12pt), stroke: (paint: fwd-blue, thickness: 0.95pt, dash: "dashed"))
          )
          // Right vertical stem going down to x_t with arrowhead
          #place(top + right, dy: 6pt,
            line(start: (0pt, 0pt), end: (0pt, 8.5pt), stroke: (paint: fwd-blue, thickness: 0.95pt, dash: "dashed"))
          )
          #place(bottom + right, dx: 3pt, dy: 0pt,
            polygon(fill: fwd-blue, stroke: none, (-3pt, -4.5pt), (3pt, -4.5pt), (0pt, 0pt))
          )
          // Centered pill label sitting cleanly on top of the horizontal dashed line
          #place(top + center, dy: 0pt,
            box(
              fill: rgb("#ffffff"),
              stroke: 0.55pt + fwd-blue,
              radius: 9pt,
              inset: (x: 8pt, y: 2.5pt),
            )[
              #text(size: 7.5pt, fill: fwd-blue)[
                *Closed-Form Direct Jump (1 Step):* $q(x_t | x_0) = cal(N)(sqrt(overline(alpha)_t) x_0, (1 - overline(alpha)_t) bold(I))$
              ]
            ]
          )
        ]
      ],
      []
    )

    #v(2pt)

    // ------------------------------------------------------------------------
    // 3. MIDDLE CHAIN: STATE NODES & BIDIRECTIONAL ARROWS
    // ------------------------------------------------------------------------
    #grid(
      columns: (1.05fr, 0.75fr, 1.05fr, 0.70fr, 1.05fr, 1.35fr, 1.05fr, 0.70fr, 1.05fr),
      align: center + horizon,
      gutter: 2pt,

      // Node x_0
      state-card([$x_0$], [Clean Data \ $x_0 tilde.op q(x_0)$], 0.0, fwd-blue),

      // Arrow x_0 <-> x_1
      [
        #text(size: 7.4pt, fill: fwd-blue)[$q(x_1 | x_0)$]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd-blue)
        #v(8pt)
        #h-arrow(dir: "left", color: rev-red)
        #v(2pt)
        #text(size: 7.4pt, fill: rev-red)[$p_theta (x_0 | x_1)$]
      ],

      // Node x_1
      state-card([$x_1$], [Slight Noise \ ($t = 1$)], 0.18, luma(130)),

      // Dots 1 <-> t-1
      [
        #text(size: 7pt, fill: fwd-blue)[$+ beta_t$]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd-blue, dashed: true)
        #v(3pt)
        #text(size: 10pt, weight: "bold", fill: luma(90))[$dots.c$]
        #v(3pt)
        #h-arrow(dir: "left", color: rev-red, dashed: true)
        #v(2pt)
        #text(size: 7pt, fill: rev-red)[$- epsilon_theta$]
      ],

      // Node x_{t-1}
      state-card([$x_(t-1)$], [Noisy State \ (Step $t - 1$)], 0.48, luma(110)),

      // Key Highlighted Transition x_{t-1} <-> x_t
      [
        #box(
          fill: fwd-bg,
          stroke: 0.45pt + fwd-blue,
          radius: 2pt,
          inset: (x: 3.5pt, y: 2pt),
        )[
          #text(size: 7.5pt, weight: "bold", fill: fwd-blue)[$q(x_t | x_(t-1))$]
        ]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd-blue)
        #v(10pt)
        #h-arrow(dir: "left", color: rev-red)
        #v(2pt)
        #box(
          fill: rev-bg,
          stroke: 0.45pt + rev-red,
          radius: 2pt,
          inset: (x: 3.5pt, y: 2pt),
        )[
          #text(size: 7.5pt, weight: "bold", fill: rev-red)[$p_theta (x_(t-1) | x_t)$]
        ]
      ],

      // Node x_t
      state-card([$x_t$], [Noisy State \ (Step $t$)], 0.68, rev-red),

      // Dots t <-> T
      [
        #text(size: 7pt, fill: fwd-blue)[$+ beta_t$]
        #v(2pt)
        #h-arrow(dir: "right", color: fwd-blue, dashed: true)
        #v(3pt)
        #text(size: 10pt, weight: "bold", fill: luma(90))[$dots.c$]
        #v(3pt)
        #h-arrow(dir: "left", color: rev-red, dashed: true)
        #v(2pt)
        #text(size: 7pt, fill: rev-red)[$- epsilon_theta$]
      ],

      // Node x_T
      state-card([$x_T$], [Pure Noise \ $cal(N)(bold(0), bold(I))$], 1.0, rev-red),
    )

    #v(7pt)

    // ------------------------------------------------------------------------
    // 4. BOTTOM PANEL: REVERSE DENOISING & U-NET NOISE PREDICTOR
    // ------------------------------------------------------------------------
    #rect(
      width: 100%,
      fill: rev-bg,
      stroke: (left: 2.8pt + rev-red, rest: 0.5pt + rgb("#eebfb8")),
      radius: 3pt,
      inset: (x: 8pt, y: 5pt),
    )[
      #grid(
        columns: (auto, 1fr),
        gutter: 8pt,
        align: (left + horizon, right + horizon),
        [
          #text(weight: "bold", size: 8.4pt, fill: rev-red)[
            REVERSE DENOISING PROCESS (Learned Sequential Chain)
          ] \
          #text(size: 7.7pt, fill: luma(45))[
            Iterative step ($T arrow.r 0$, no skip): $p_theta (x_(t-1) | x_t) = cal(N)(x_(t-1); mu_theta (x_t, t), sigma_t^2 bold(I))$
          ]
        ],
        [
          #box(
            fill: unet-bg,
            stroke: 0.65pt + unet-purple,
            radius: 3pt,
            inset: (x: 7pt, y: 4pt),
          )[
            #align(center)[
              #text(size: 7.5pt, weight: "bold", fill: unet-purple)[U-Net Noise Predictor $epsilon_theta (x_t, t)$] \
              #v(1pt)
              #text(size: 7.6pt)[
                $L_("simple") = bb(E)_(t, x_0, epsilon) [ norm(epsilon - epsilon_theta (x_t, t))^2 ]$
              ]
            ]
          ]
        ]
      )
    ]
  ]
}

// Render when compiled as a standalone file
#fig-ddpm-chain()
