// Generated from saved JSON; run experiments/Scripts/step4_compare_benchmark.py.
#set page(width: 16.6cm, height: auto, margin: 4pt)
#set text(font: "New Computer Modern", size: 9pt)
#set par(first-line-indent: 0pt, justify: false)
#let panel(title, unit, a, b, aw, bw) = block(
  width: 100%, stroke: 0.5pt + luma(65%), radius: 3pt, inset: 7pt,
)[
  #text(weight: "bold")[#title]
  #linebreak()
  #text(size: 8pt, fill: luma(35%))[#unit]
  #v(5pt)
  #grid(columns: (38pt, 1fr, 43pt), gutter: 4pt, align: horizon,
    [Pixel], rect(width: aw, height: 8pt, fill: luma(25%), stroke: none), [#a],
    [Latent], rect(width: bw, height: 8pt, fill: luma(60%), stroke: none), [#b],
  )
]
#grid(columns: (1fr,1fr), gutter: 8pt,
  panel("Counted U-Net arithmetic", "million FLOPs / evaluation", "117.87", "8.31", 100.000%, 7.048%),
  panel("Mean denoiser epoch", "seconds / epoch", "9.42", "5.74", 100.000%, 60.881%),
  panel("Sum of timed training epochs", "seconds; latent includes VAE", "471.2", "458.6", 100.000%, 97.321%),
  panel("Recorded sampling trajectory", "seconds; instrumentation differs", "1.592", "1.715", 92.828%, 100.000%),
)
