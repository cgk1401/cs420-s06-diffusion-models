// Existing sample images; no new inference. See data/provenance.json.
#set page(width: 16.6cm, height: auto, margin: 3pt)
#set text(font: "New Computer Modern", size: 9pt)
#set par(first-line-indent: 0pt)
#let panel(n, label) = align(center)[
  #box(width: 100%, height: 25pt)[#align(center + horizon)[#label]]
  #v(4pt)
  #image("data/fig_exp_ldm_latent_and_noise-" + (if n < 10 { "0" + str(n) } else { str(n) }) + ".png", width: 3cm)
]
#grid(columns: (1fr,)*3, gutter: 6pt,
  panel(0, [Input $x_0$: $1 times 28 times 28$]),
  panel(1, [Latent: $4 times 7 times 7$ \ Channel mean]),
  panel(2, [Reconstruction $tilde(x)$ \ $1 times 28 times 28$]),
)
