"""
Interactive Demonstration: Spatial vs. Spectral Domain Analysis
---------------------------------------------------------------
Generates sample images and produces a 6-panel scientific comparison
demonstrating natural 1/f^2 energy falloff vs. generative periodic upsampling artifacts.
Saves the resulting figure to results/figures/spectral_comparison.png.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.spectral import plot_spectral_comparison
import numpy as np
from PIL import Image

# Ensure output directory exists
fig_dir = Path("results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("data/interim")
scratch_dir.mkdir(parents=True, exist_ok=True)

# Generate synthetic demonstration images if real images are not available
real_demo_path = scratch_dir / "demo_real.jpg"
fake_demo_path = scratch_dir / "demo_fake.jpg"

np.random.seed(42)
size = 256

# 1. Simulate Natural Image: Smooth gradient with 1/f spatial correlation
y, x = np.mgrid[-1:1:256j, -1:1:256j]
natural_pattern = np.sin(x * 3 + y * 2) * 50 + 128
natural_pattern += np.random.normal(0, 5, (size, size))
natural_rgb = np.stack([natural_pattern, natural_pattern * 0.9 + 20, natural_pattern * 0.8 + 40], axis=-1)
natural_rgb = np.clip(natural_rgb, 0, 255).astype(np.uint8)
Image.fromarray(natural_rgb).save(real_demo_path)

# 2. Simulate AI/GAN Image: Similar visual scene but with subtle periodic transposed convolution grid artifacts
fake_pattern = np.copy(natural_pattern)
# Introduce subtle periodic checkerboard / upsampling frequency grid (characteristic of ConvTranspose2d)
grid_artifact = (np.sin(x * 64) * np.sin(y * 64)) * 12
fake_pattern = fake_pattern + grid_artifact
fake_rgb = np.stack([fake_pattern, fake_pattern * 0.9 + 20, fake_pattern * 0.8 + 40], axis=-1)
fake_rgb = np.clip(fake_rgb, 0, 255).astype(np.uint8)
Image.fromarray(fake_rgb).save(fake_demo_path)

output_fig = fig_dir / "spectral_comparison.png"
print(f"Generating spectral comparison visualization...")
plot_spectral_comparison(str(real_demo_path), str(fake_demo_path), str(output_fig))
print(f"Success! Inspection figure saved to {output_fig}")
