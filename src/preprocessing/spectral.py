"""
Spectral Analysis & Frequency-Domain Feature Extraction Module
--------------------------------------------------------------
Implements 2D Fast Fourier Transform (2D FFT), centered log-magnitude
spectrum computation, and 1D Azimuthal / Radial averaging.

Scientific Background:
Generative models (GANs, Diffusion models, VAEs) use transposed convolutions
and upsampling layers that introduce subtle, periodic grid artifacts.
While natural photos exhibit a power spectrum decay following P(f) ~ 1/f^alpha (alpha ~ 2),
synthetic images display distinct periodic spikes and high-frequency discrepancies.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T


def compute_fft_2d_numpy(image_gray: np.ndarray) -> np.ndarray:
    """
    Computes the 2D Fast Fourier Transform of a 2D grayscale image using NumPy.
    Returns the centered 2D complex spectrum.
    
    Parameters:
        image_gray: 2D numpy array with shape (H, W)
    Returns:
        fft_shift: 2D complex numpy array centered at DC frequency (0, 0)
    """
    if image_gray.ndim == 3:
        # Convert RGB to Grayscale using luminosity weights
        image_gray = np.dot(image_gray[..., :3], [0.2989, 0.5870, 0.1140])
    
    # 2D Discrete Fourier Transform
    fft2 = np.fft.fft2(image_gray)
    # Shift zero-frequency component to the center of the spectrum
    fft_shift = np.fft.fftshift(fft2)
    return fft_shift


def get_log_magnitude_spectrum_numpy(image_gray: np.ndarray) -> np.ndarray:
    """
    Computes normalized log magnitude spectrum: log(1 + |F(u, v)|).
    Output is scaled to [0, 1].
    """
    fft_shift = compute_fft_2d_numpy(image_gray)
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)  # log(1 + magnitude) prevents log(0)
    
    # Min-max normalization for visual and numerical stability
    min_val = log_magnitude.min()
    max_val = log_magnitude.max()
    if max_val - min_val > 1e-8:
        normalized = (log_magnitude - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(log_magnitude)
    return normalized.astype(np.float32)


def compute_radial_profile(log_magnitude: np.ndarray) -> np.ndarray:
    """
    Computes the 1D Azimuthal (radially averaged) power spectrum profile.
    Averages pixel values at concentric rings of radius r from the center.
    
    Parameters:
        log_magnitude: 2D numpy array (centered frequency spectrum)
    Returns:
        radial_profile: 1D numpy array representing energy vs spatial frequency
    """
    h, w = log_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - center_x, y - center_y).astype(int)
    
    # Compute the average value in each radial bin
    max_r = min(center_x, center_y)
    r_flat = r.ravel()
    mag_flat = log_magnitude.ravel()
    
    tbin = np.bincount(r_flat, mag_flat)
    nr = np.bincount(r_flat)
    radial_profile = tbin / np.maximum(nr, 1)
    
    return radial_profile[:max_r].astype(np.float32)


def compute_fft_2d_torch(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    PyTorch GPU/CPU batch implementation of 2D FFT Log-Magnitude Spectrum.
    
    Parameters:
        image_tensor: Tensor of shape (B, C, H, W) or (C, H, W), range [0, 1] or normalized.
    Returns:
        log_mag: Tensor of shape (B, 1, H, W), values scaled to [0, 1].
    """
    squeeze_batch = False
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
        squeeze_batch = True
        
    B, C, H, W = image_tensor.shape
    
    # Convert to grayscale if 3 channels
    if C == 3:
        # Standard Luminance coefficients
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image_tensor.device).view(1, 3, 1, 1)
        gray = torch.sum(image_tensor * weights, dim=1, keepdim=True)  # (B, 1, H, W)
    else:
        gray = image_tensor[:, :1, :, :]
        
    # 2D FFT on the spatial dimensions (dim=(-2, -1))
    fft2 = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft2, dim=(-2, -1))
    
    # Log magnitude
    mag = torch.abs(fft_shift)
    log_mag = torch.log1p(mag)
    
    # Batch-wise min-max normalization
    min_val = log_mag.amin(dim=(-2, -1), keepdim=True)
    max_val = log_mag.amax(dim=(-2, -1), keepdim=True)
    denom = torch.clamp(max_val - min_val, min=1e-8)
    norm_log_mag = (log_mag - min_val) / denom
    
    if squeeze_batch:
        norm_log_mag = norm_log_mag.squeeze(0)
        
    return norm_log_mag


class SpectralTransform(nn.Module):
    """
    PyTorch Module / Transform that can be embedded directly inside
    Dataset transforms or inside the forward pass of the model.
    """
    def __init__(self, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        self.resize = T.Resize(target_size, antialias=True)
        
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Takes RGB image tensor (B, 3, H, W) and outputs (B, 1, H_target, W_target)
        representing the normalized 2D FFT magnitude spectrum.
        """
        if img_tensor.shape[-2:] != self.target_size:
            img_tensor = self.resize(img_tensor)
        return compute_fft_2d_torch(img_tensor)


def plot_spectral_comparison(real_image_path: str, fake_image_path: str, output_path: str = "spectral_comparison.png"):
    """
    Generates a 6-panel scientific comparison figure:
    - Real Image (Spatial) vs Real 2D FFT Magnitude vs Real Radial Energy Profile
    - AI Image (Spatial) vs AI 2D FFT Magnitude vs AI Radial Energy Profile
    """
    import matplotlib.pyplot as plt
    
    real_img = Image.open(real_image_path).convert("RGB")
    fake_img = Image.open(fake_image_path).convert("RGB")
    
    real_arr = np.array(real_img.resize((256, 256)))
    fake_arr = np.array(fake_img.resize((256, 256)))
    
    real_fft = get_log_magnitude_spectrum_numpy(real_arr)
    fake_fft = get_log_magnitude_spectrum_numpy(fake_arr)
    
    real_radial = compute_radial_profile(real_fft)
    fake_radial = compute_radial_profile(fake_fft)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Real row
    axes[0, 0].imshow(real_arr)
    axes[0, 0].set_title("Real Image (Spatial Domain)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    
    im0 = axes[0, 1].imshow(real_fft, cmap="inferno")
    axes[0, 1].set_title("Real 2D FFT Spectrum (Centered)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    fig.colorbar(im0, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].plot(real_radial, color="forestgreen", lw=2, label="Real (1/f^alpha decay)")
    axes[0, 2].set_title("Radial Power Spectrum", fontsize=12, fontweight="bold")
    axes[0, 2].set_xlabel("Spatial Frequency (r)")
    axes[0, 2].set_ylabel("Normalized Energy")
    axes[0, 2].grid(True, linestyle="--", alpha=0.6)
    axes[0, 2].legend()
    
    # Fake row
    axes[1, 0].imshow(fake_arr)
    axes[1, 0].set_title("AI / Synthetic Image (Spatial Domain)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    
    im1 = axes[1, 1].imshow(fake_fft, cmap="inferno")
    axes[1, 1].set_title("AI 2D FFT Spectrum (Periodic Peaks)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].plot(fake_radial, color="crimson", lw=2, label="AI Generated (Abnormal high-freq)")
    axes[1, 2].set_title("Radial Power Spectrum", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Spatial Frequency (r)")
    axes[1, 2].set_ylabel("Normalized Energy")
    axes[1, 2].grid(True, linestyle="--", alpha=0.6)
    axes[1, 2].legend()
    
    plt.suptitle("Spatial vs Frequency Domain Analysis (Deepfake / AI Detection)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {output_path}")
