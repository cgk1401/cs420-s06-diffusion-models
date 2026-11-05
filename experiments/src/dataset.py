"""
Dataset utilities for 28x28 MNIST images normalized to [-1.0, 1.0].
Provides both DataLoader access and a fixed canonical 28x28 sample image (digit '3')
so that Section 3.2 (1D worked example) and Section 4 (28x28 experiments) share
the exact same pixel values.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def scale_to_minus1_1(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Map [0, 1] or [0, 255] intensity values into [-1.0, 1.0]."""
    if isinstance(x, np.ndarray):
        if x.max() > 1.5:
            return (x.astype(np.float32) / 127.5) - 1.0
        return (x.astype(np.float32) * 2.0) - 1.0
    if x.max() > 1.5:
        return (x.float() / 127.5) - 1.0
    return (x.float() * 2.0) - 1.0


def unscale_to_0_1(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Map [-1.0, 1.0] tensor/array back to [0.0, 1.0] numpy array for plotting."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def get_mnist_datasets(download: bool = True):
    """Load train and test MNIST datasets normalized to [-1.0, 1.0] in shape [1, 28, 28]."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # maps [0, 1] -> [-1, 1]
    ])
    train_ds = datasets.MNIST(root=str(DATA_ROOT), train=True, download=download, transform=transform)
    test_ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=download, transform=transform)
    return train_ds, test_ds


def get_mnist_dataloader(batch_size: int = 128, max_samples: int | None = 12000, seed: int = 42):
    """Return a DataLoader for training on 28x28 MNIST."""
    train_ds, _ = get_mnist_datasets(download=True)
    if max_samples is not None and max_samples < len(train_ds):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(train_ds), size=max_samples, replace=False).tolist()
        train_ds = Subset(train_ds, indices)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)


def get_canonical_digit_sample(digit: int = 3) -> tuple[torch.Tensor, int, tuple[int, int]]:
    """
    Return a deterministic, clean 28x28 MNIST sample of `digit` (default: digit '3'),
    its label, and a canonical foreground pixel coordinate (row, col) on the digit stroke.
    """
    _, test_ds = get_mnist_datasets(download=True)
    for idx in range(len(test_ds)):
        img, label = test_ds[idx]
        if int(label) == digit:
            # Find a representative bright stroke pixel near the center (around row 13, col 14)
            # that has high intensity (> 0.5 in [-1, 1], but not saturated at 1.0 if possible)
            arr = img[0].numpy()
            candidates = []
            for r in range(10, 18):
                for c in range(10, 18):
                    if 0.45 <= arr[r, c] <= 0.96:
                        candidates.append((abs(r - 13) + abs(c - 14), r, c))
            if candidates:
                candidates.sort()
                _, best_r, best_c = candidates[0]
            else:
                best_r, best_c = 13, 14
            return img, int(label), (best_r, best_c)
    raise RuntimeError(f"Digit {digit} not found in MNIST test set.")


if __name__ == "__main__":
    # Quick test of canonical digit sample
    img, label, (r, c) = get_canonical_digit_sample(digit=3)
    print(f"Canonical digit sample: label={label}, pixel=({r}, {c}), value={img[0, r, c]:.3f}")
