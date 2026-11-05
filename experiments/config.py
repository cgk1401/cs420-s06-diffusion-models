from pathlib import Path

# Base paths
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent


OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

for _d in (OUTPUTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Global random seed for complete reproducibility
SEED = 42

# Image & Latent dimensions
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
LATENT_SIZE = 7          # Downsampling factor f = 28 / 7 = 4 (spatial area reduced 16x)
LATENT_CHANNELS = 4      # 4 x 7 x 7 = 196 elements (vs. 1 x 28 x 28 = 784 elements)

# DDPM Linear Variance Schedule (Ho et al., 2020)
T = 1000
BETA_START = 1e-4        # beta_1 = 0.0001
BETA_END = 0.02          # beta_T = 0.02

# Five noise levels for Task 4a (0-indexed indices corresponding to t = 50, 200, 400, 700, 1000)
FORWARD_STEPS_5 = [49, 199, 399, 699, 999]
FORWARD_LABELS_5 = ["t = 50", "t = 200", "t = 400", "t = 700", "t = 1000"]

# Reverse denoising observation steps for Task 4b
REVERSE_STEPS_SHOW = [999, 699, 399, 199, 49, 0]
REVERSE_LABELS_SHOW = ["t = 1000", "t = 700", "t = 400", "t = 200", "t = 50", "t = 0"]

