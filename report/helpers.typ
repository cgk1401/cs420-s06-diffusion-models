#import "config.typ": show-guidelines

#let guideline(tag, body) = {
  if show-guidelines {
    block(
      width: 100%,
      fill: rgb("#f2f6fc"),
      stroke: (left: 2.5pt + rgb("#2b579a"), rest: 0.4pt + rgb("#c5d5ea")),
      inset: 9pt,
      radius: 2pt,
      above: 0.7em,
      below: 0.85em,
    )[
      #set par(first-line-indent: 0pt, leading: 0.55em, spacing: 0.5em)
      #set text(size: 9.2pt, fill: luma(35))
      #text(weight: "bold", fill: rgb("#1e4278"))[\[WRITING GUIDE — #tag\]]
      #v(2pt)
      #body
    ]
  }
}

#let fig-placeholder(label-text, height: 3.4cm) = {
  rect(
    width: 100%,
    height: height,
    fill: luma(246),
    stroke: (paint: luma(160), thickness: 0.7pt, dash: "dashed"),
    radius: 3pt,
    inset: 12pt,
  )[
    #set par(first-line-indent: 0pt)
    #align(center + horizon)[
      #text(size: 9.2pt, fill: luma(85))[
        *\[FIGURE / DIAGRAM PLACEHOLDER\]* \
        #v(3pt)
        #label-text
      ]
    ]
  ]
}
