// ============================================================================
// CS420 SEMINAR RESEARCH REPORT TEMPLATE [S06] — SINGLE-COLUMN ACADEMIC PAPER
// Compatible with typst.app (multi-user real-time collaboration) & Typst CLI
// ============================================================================

// ============================================================================
// DOCUMENT & ACADEMIC PAPER STYLING (SINGLE-COLUMN A4 RESEARCH FORMAT)
// ============================================================================
#set document(
  title: "Diffusion Models: From Adding Noise to Generating Images",
  author: ("Author Name 1", "Author Name 2", "Author Name 3"),
)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.2cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8.5pt, fill: luma(110))
      CS420 Seminar Research Report — Topic [S06]: Diffusion Models
      #h(1fr)
      _From Adding Noise to Generating Images_
      #v(-5pt)
      #line(length: 100%, stroke: 0.4pt + luma(190))
    ]
  },
)

#set text(
  font: ("New Computer Modern", "Linux Libertine"),
  size: 10.5pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.62em,
  spacing: 0.85em,
  first-line-indent: 1.2em,
)

#set heading(numbering: "1.1")

#show heading.where(level: 1): it => block(
  above: 1.45em,
  below: 0.7em,
  text(size: 12.5pt, weight: "bold", it),
)

#show heading.where(level: 2): it => block(
  above: 1.15em,
  below: 0.55em,
  text(size: 11pt, weight: "bold", it),
)

#set math.equation(numbering: "(1)", supplement: [Eq.])
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(size: 9.2pt)

#include "sections/title-and-abstract.typ"

#include "sections/introduction.typ"

#include "sections/background.typ"

#include "sections/method.typ"

#include "sections/results.typ"

#include "sections/conclusion.typ"

#pagebreak()
#bibliography("references.bib", title: "References", style: "ieee")

#include "sections/appendix.typ"
