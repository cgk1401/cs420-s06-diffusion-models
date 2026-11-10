"""Generate conservative report assets from immutable saved experiment records.

No training or inference is performed. No PyTorch or plotting dependency is needed.
The chart source is Typst; build_report.py compiles it with the rest of the report.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "experiments" / "outputs" / "tables"


def generate_assets():
    pixel = json.loads((TABLES / "ddpm_pixel_stats.json").read_text(encoding="utf-8"))
    latent = json.loads((TABLES / "ldm_latent_stats.json").read_text(encoding="utf-8"))
    vae = json.loads((TABLES / "vae_stats.json").read_text(encoding="utf-8"))
    panels = [
        ("Counted U-Net arithmetic", "million FLOPs / evaluation", pixel["flops_per_step"] / 1e6, latent["flops_per_step"] / 1e6, 2),
        ("Mean denoiser epoch", "seconds / epoch", pixel["avg_epoch_time_sec"], latent["avg_epoch_time_sec"], 2),
        ("Sum of timed training epochs", "seconds; latent includes VAE", pixel["total_train_time_sec"], latent["total_train_time_sec"] + vae["total_train_time_sec"], 1),
        ("Recorded sampling trajectory", "seconds; instrumentation differs", pixel["sampling_time_1000_steps_sec"], latent["sampling_time_1000_steps_sec"], 3),
    ]
    source = '''// Generated from saved JSON; run experiments/Scripts/step4_compare_benchmark.py.
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
'''
    for title, unit, a, b, precision in panels:
        maximum = max(a, b)
        source += f'  panel("{title}", "{unit}", "{a:.{precision}f}", "{b:.{precision}f}", {a / maximum * 100:.3f}%, {b / maximum * 100:.3f}%),\n'
    source += ")\n"
    target = ROOT / "typst" / "figures" / "fig_speed_flops_comparison.typ"
    target.write_text(source, encoding="utf-8")
    ratio = pixel["flops_per_step"] / latent["flops_per_step"]
    md = "# Recorded MNIST comparison\n\n"
    md += "Source: checked-in ddpm_pixel_stats.json, ldm_latent_stats.json and vae_stats.json. Single-run descriptive observations; learning rates, preprocessing and sampling instrumentation differ.\n\n"
    md += "| Metric | Pixel | Latent |\n|---|---:|---:|\n"
    for title, unit, a, b, precision in panels:
        md += f"| {title} ({unit}) | {a:.{precision}f} | {b:.{precision}f} |\n"
    md += f"\nCounted arithmetic ratio: {ratio:.2f}x, using convolution/linear MACs only and 2 FLOPs/MAC. Other operators and VAE decode are excluded.\n"
    md += "\nTraining sums exclude data preparation and latent-cache construction. Pixel sampling records intermediate CPU snapshots; latent sampling includes decoding. These are not matched latency measurements.\n"
    md += "\nBoth models use additive class embeddings and single-token cross-attention. No spatial token-selection mechanism is established by the class examples. No distributional quality ranking is claimed.\n"
    md += "\nMemory comparison is omitted: legacy peak-allocation fields do not identify standardized workloads and may have been refreshed during evaluation. Raw JSON is preserved unchanged.\n"
    (TABLES / "pixel_vs_latent_benchmark.md").write_text(md, encoding="utf-8")
    return panels


def main():
    generate_assets()
    print("Generated benchmark Markdown and Typst chart source. Run python tools/build_report.py from the repository root to compile the chart and report.")


if __name__ == "__main__":
    main()
