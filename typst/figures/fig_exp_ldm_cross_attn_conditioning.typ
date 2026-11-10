// Existing conditioned samples; later stochastic draws were not shared.
#set page(width: 16.6cm, height: auto, margin: 3pt)
#set text(font: "New Computer Modern", size: 9pt)
#set par(first-line-indent: 0pt)
#let panel(n, label) = align(center)[
  #label
  #v(4pt)
  #image("data/fig_exp_ldm_cross_attn_conditioning-" + (if n < 10 { "0" + str(n) } else { str(n) }) + ".png", width: 2.65cm)
]
#align(center)[Class labels enter through additive embeddings and single-token cross-attention.]
#v(5pt)
#grid(columns: (1fr,)*4, gutter: 5pt,
  panel(0, [Shared initial $z_T$]), panel(1, [$y=0$]), panel(2, [$y=3$]), panel(3, [$y=7$]),
)
