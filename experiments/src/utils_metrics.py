"""
Publication-grade figure rendering (300 DPI + vector PDF) and computational
benchmarking utilities (FLOPs/MACs, latency, memory, PSNR/MSE).
"""

from pathlib import Path
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Select the best available compute device.
    Returns CUDA if a GPU is available (and `prefer_cuda` is True), otherwise CPU.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device | str) -> str:
    """Human-readable device description for logging (e.g. 'cuda (NVIDIA ...)' or 'cpu')."""
    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return "cpu"


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_single_step_macs(model: nn.Module, input_shape: tuple[int, int, int, int], device: str = "cpu") -> int:
    """
    Accurately count Multiply-Accumulate operations (MACs) for a single U-Net forward pass
    using forward hooks on Conv2d, ConvTranspose2d, and Linear layers.
    Note: 1 MAC = 2 FLOPs.
    """
    total_macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inp, out):
        nonlocal total_macs
        out_t = out[0] if isinstance(out, tuple) else out
        b, c_out, h_out, w_out = out_t.shape
        c_in = module.in_channels // module.groups
        k_h, k_w = module.kernel_size
        macs = b * c_out * h_out * w_out * c_in * k_h * k_w
        total_macs += macs

    def linear_hook(module: nn.Linear, inp, out):
        nonlocal total_macs
        inp_t = inp[0]
        num_instances = inp_t.numel() // module.in_features
        total_macs += num_instances * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    x_dummy = torch.randn(input_shape, device=device)
    t_dummy = torch.zeros((input_shape[0],), dtype=torch.long, device=device)
    y_dummy = torch.zeros((input_shape[0],), dtype=torch.long, device=device)

    with torch.no_grad():
        model(x_dummy, t_dummy, y_dummy)

    for h in hooks:
        h.remove()

    return int(total_macs)


def compute_mse_psnr(x_true_01: np.ndarray, x_recon_01: np.ndarray) -> tuple[float, float]:
    """Compute Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR in dB) on [0, 1] images."""
    mse = float(np.mean((x_true_01 - x_recon_01) ** 2))
    if mse < 1e-10:
        return 0.0, 100.0
    psnr = 10.0 * math.log10(1.0 / mse)
    return mse, psnr


def save_figure_both(fig: plt.Figure, png_path: Path, also_pdf: bool = True):
    """Save figure at 300 DPI PNG and vector PDF simultaneously."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, facecolor="white")
    if also_pdf:
        pdf_path = png_path.with_suffix(".pdf")
        fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
