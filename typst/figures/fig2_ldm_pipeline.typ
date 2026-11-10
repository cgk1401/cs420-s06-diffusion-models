// KL-regularized LDM schematic, drawn by the authors from Rombach et al. (2022).
#set page(width: 16.6cm, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt, lang: "en")
#set par(first-line-indent: 0pt, justify: false, leading: 0.45em)
#let card(body) = block(width: 100%, inset: 7pt, radius: 3pt, stroke: 0.6pt + luma(140), fill: luma(97%), body)
#card[
  #text(weight: "bold")[Stage 1: Train the autoencoder]
  #v(5pt)
  #align(center)[$x arrow.r cal(E) arrow.r (mu,sigma) arrow.r z=mu+sigma dot.o eta arrow.r cal(D) arrow.r tilde(x)$]
  #v(4pt)
  #align(center)[Reconstruction loss + KL regularization; $eta ~ cal(N)(0,bold(I))$]
]
#v(7pt)
#card[
  #text(weight: "bold")[Stage 2: Freeze the autoencoder; train latent diffusion]
  #v(5pt)
  #align(center)[$x arrow.r cal(E) arrow.r z_0 arrow.r z_t arrow.r epsilon_theta(z_t,t,tau_theta(y))$]
  #v(4pt)
  #align(center)[Add known Gaussian noise; optimize the noise-prediction error.]
]
#v(7pt)
#card[
  #text(weight: "bold")[Generation: No input image is required]
  #v(5pt)
  #align(center)[$z_T ~ cal(N)(0,bold(I)) quad arrow.r quad z_(T-1) quad arrow.r quad dots quad arrow.r quad z_0^"gen" arrow.r cal(D) arrow.r x_"gen"$]
  #v(4pt)
  #align(center)[Repeat: predict noise, update latent, and add reverse noise if $t>1$.]
  #align(center)[Condition $y arrow.r tau_theta(y)$ supplies keys/values to the denoiser.]
]
