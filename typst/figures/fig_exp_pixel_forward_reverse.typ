// Existing image samples extracted losslessly; see data/provenance.json.
#set page(width: 16.6cm, height: auto, margin: 3pt)
#set text(font: "New Computer Modern", size: 9pt)
#set par(first-line-indent: 0pt)
#let panel(n, label) = align(center)[
  #text(size: 9pt)[#label]
  #v(3pt)
  #box(width: 2.35cm, height: 2.35cm)[
    #image("data/fig_exp_pixel_forward_reverse-" + (if n < 10 { "0" + str(n) } else { str(n) }) + ".png", width: 100%, height: 100%)
    #place(top + left, dx: 50%, dy: 35.714%)[
      #rect(width: 2.35cm / 28, height: 2.35cm / 28, stroke: 0.5pt + rgb("c62828"), fill: none)
    ]
  ]
]
#text(weight: "bold")[Forward marginals from the selected test image]
#v(3pt)
#grid(columns: (1fr,)*6, gutter: 3pt,
  panel(0, [$t=0$]), panel(1, [$t=50$]), panel(2, [$t=200$]),
  panel(3, [$t=400$]), panel(4, [$t=700$]), panel(5, [$t=1000$]),
)
#v(7pt)
#text(weight: "bold")[Independent reverse generation, class $y=3$]
#v(3pt)
#grid(columns: (1fr,)*6, gutter: 3pt,
  panel(6, [$t=1000$]), panel(7, [$t=699$]), panel(8, [$t=399$]),
  panel(9, [$t=199$]), panel(10, [$t=49$]), panel(11, [$t=0$]),
)
