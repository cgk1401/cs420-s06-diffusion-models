"""
Publication-grade figure rendering (300 DPI + vector PDF) and computational
benchmarking utilities (FLOPs/MACs, latency, memory, PSNR/MSE).
"""

from pathlib import Path
import matplotlib.pyplot as plt

# Configure publication-style matplotlib typography matching Typst Computer Modern
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def save_figure_both(fig: plt.Figure, png_path: Path, also_pdf: bool = True):
    """Save figure at 300 DPI PNG and vector PDF simultaneously."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor="white")
    if also_pdf:
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
