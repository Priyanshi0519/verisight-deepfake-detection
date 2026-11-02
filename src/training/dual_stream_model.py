"""
Dual-Stream Deep Learning Model for AI Image / Deepfake Detection
-----------------------------------------------------------------
Combines a Spatial Stream (RGB pixel domain) and a Spectral Stream
(2D FFT frequency domain) into a unified feature-fusion network.

Features:
- Spatial Stream: ResNet (or DenseNet) for texture, blending, and semantic cues.
- Spectral Stream: Custom Convolutional Frequency Encoder for periodic upsampling artifacts.
- Fusion Classifier: Multi-Layer Perceptron (MLP) with dropout.
- Explainability: Outputs individual Spatial & Spectral confidence scores alongside the final ensemble probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.preprocessing.spectral import compute_fft_2d_torch
except (ImportError, ModuleNotFoundError):
    from ..preprocessing.spectral import compute_fft_2d_torch


class SpectralCNNEncoder(nn.Module):
    """
    Dedicated Convolutional Network for processing 2D FFT Log-Magnitude Spectra.
    Input shape: (B, 1, H, W) -> Output shape: (B, out_features)
    """
    def __init__(self, out_features: int = 256):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Capture fine high-frequency grid lines
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56
            
            # Block 3: Mid-frequency structures
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            
            # Block 4: Global spectral energy distribution
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 256, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.fc(feat)


class DualStreamDetector(nn.Module):
    """
    Dual-Stream AI-Generated & Deepfake Image Detector.
    
    Spatial Branch: Pretrained ResNet50 or ResNet18
    Spectral Branch: SpectralCNNEncoder (2D FFT Magnitude)
    Fusion Head: Concatenated dense classification layers
    """
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone_name = backbone
        
        # 1. Spatial Stream
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            spatial_features_dim = resnet.fc.in_features  # 2048
            resnet.fc = nn.Identity()
            self.spatial_encoder = resnet
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            spatial_features_dim = resnet.fc.in_features  # 512
            resnet.fc = nn.Identity()
            self.spatial_encoder = resnet
        elif backbone == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            densenet = models.densenet121(weights=weights)
            spatial_features_dim = densenet.classifier.in_features  # 1024
            densenet.classifier = nn.Identity()
            self.spatial_encoder = densenet
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # 2. Spectral Stream
        spectral_features_dim = 256
        self.spectral_encoder = SpectralCNNEncoder(out_features=spectral_features_dim)
        
        # Auxiliary heads for explainability
        self.spatial_head = nn.Linear(spatial_features_dim, num_classes)
        self.spectral_head = nn.Linear(spectral_features_dim, num_classes)
        self.spatial_aux_head = self.spatial_head
        self.spectral_aux_head = self.spectral_head
        
        # 3. Fusion Head
        total_dim = spatial_features_dim + spectral_features_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x_rgb: torch.Tensor, return_breakdown: bool = False):
        """
        Forward pass.
        Parameters:
            x_rgb: RGB image tensor of shape (B, 3, H, W)
            return_breakdown: If True, returns dict with spatial, spectral, and fusion predictions.
        """
        # Spatial stream extraction
        spatial_emb = self.spatial_encoder(x_rgb)
        
        # Spectral stream extraction via on-the-fly 2D FFT
        spectral_input = compute_fft_2d_torch(x_rgb)  # (B, 1, H, W)
        spectral_emb = self.spectral_encoder(spectral_input)
        
        # Concatenate embeddings
        fused_emb = torch.cat([spatial_emb, spectral_emb], dim=1)
        fused_logits = self.fusion_head(fused_emb)
        
        if return_breakdown:
            spatial_logits = self.spatial_head(spatial_emb)
            spectral_logits = self.spectral_head(spectral_emb)
            
            return {
                "logits": fused_logits,
                "spatial_logits": spatial_logits,
                "spectral_logits": spectral_logits,
                "spatial_emb": spatial_emb,
                "spectral_emb": spectral_emb,
                "spectral_map": spectral_input
            }
            
        return fused_logits

    def predict_probabilities(self, x_rgb: torch.Tensor, temperature: float = 2.0) -> list:
        """
        Inference helper returning calibrated softmax probabilities for
        fusion, spatial-only, and spectral-only streams.
        Dynamically detects fake/AI class index from class_to_idx.
        Applies temperature scaling and logit bounding to prevent numerical saturation
        on high-resolution out-of-distribution web images.
        """
        self.eval()
        fake_idx = 0
        if hasattr(self, "class_to_idx") and self.class_to_idx:
            for k, v in self.class_to_idx.items():
                if "fake" in k.lower() or "ai" in k.lower() or "synth" in k.lower():
                    fake_idx = v
                    break

        with torch.no_grad():
            outputs = self.forward(x_rgb, return_breakdown=True)
            
            # Apply temperature scaling & logit bounding to prevent saturation
            scaled_fusion_logits = torch.clamp(outputs["logits"], -12.0, 12.0) / temperature
            scaled_spatial_logits = torch.clamp(outputs["spatial_logits"], -12.0, 12.0) / temperature
            scaled_spectral_logits = torch.clamp(outputs["spectral_logits"], -12.0, 12.0) / temperature
            
            fusion_probs = F.softmax(scaled_fusion_logits, dim=1)
            spatial_probs = F.softmax(scaled_spatial_logits, dim=1)
            spectral_probs = F.softmax(scaled_spectral_logits, dim=1)
            
            # Probability of being AI / Fake
            ai_prob = fusion_probs[:, fake_idx].tolist()
            spatial_ai_prob = spatial_probs[:, fake_idx].tolist()
            spectral_ai_prob = spectral_probs[:, fake_idx].tolist()
            
            results = []
            for i in range(len(ai_prob)):
                prob = ai_prob[i]
                label = "AI-Generated" if prob >= 0.50 else "Real"
                confidence = prob if prob >= 0.50 else (1.0 - prob)
                
                results.append({
                    "label": label,
                    "is_ai": prob >= 0.50,
                    "confidence_pct": round(confidence * 100, 1),
                    "ai_probability": round(prob, 4),
                    "spatial_ai_probability": round(spatial_ai_prob[i], 4),
                    "spectral_ai_probability": round(spectral_ai_prob[i], 4)
                })
                
            return results


class ColabSpectralCNNEncoder(nn.Module):
    """Spectral encoder corresponding to notebook 07 sequential layout."""
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ColabDualStreamDetector(nn.Module):
    """
    Dual-stream architecture used in 07_high_accuracy_gravex_training.ipynb.
    Spatial Stream: nn.Sequential(*list(models.resnet50().children())[:-1])
    Spectral Stream: ColabSpectralCNNEncoder
    Provides strict 100% parameter matching for ai_generative_stream_best.pth.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        base = models.resnet50(weights=None)
        self.spatial_encoder = nn.Sequential(*list(base.children())[:-1])
        self.spectral_encoder = ColabSpectralCNNEncoder(out_features=256)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(2048 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.spatial_head = nn.Linear(2048, num_classes)
        self.spectral_head = nn.Linear(256, num_classes)

    @property
    def spatial_aux_head(self):
        return self.spatial_head

    @property
    def spectral_aux_head(self):
        return self.spectral_head

    def extract_fft_spectrum(self, x_rgb: torch.Tensor) -> torch.Tensor:
        gray = 0.2989 * x_rgb[:, 0:1] + 0.5870 * x_rgb[:, 1:2] + 0.1140 * x_rgb[:, 2:3]
        fft = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft)
        mag = torch.log1p(torch.abs(fft_shifted))
        b = mag.shape[0]
        min_v = mag.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        max_v = mag.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        return (mag - min_v) / (max_v - min_v + 1e-8)

    def forward(self, x: torch.Tensor, return_breakdown: bool = False):
        f_spatial = self.spatial_encoder(x).flatten(1)
        x_fft = self.extract_fft_spectrum(x)
        f_spectral = self.spectral_encoder(x_fft)
        
        fused = torch.cat([f_spatial, f_spectral], dim=1)
        logits = self.fusion_head(fused)
        
        if return_breakdown:
            return {
                'logits': logits,
                'spatial_logits': self.spatial_head(f_spatial),
                'spectral_logits': self.spectral_head(f_spectral)
            }
        return logits

    def predict_proba(self, x_rgb: torch.Tensor, temperature: float = 1.5):
        self.eval()
        fake_idx = 0
        if hasattr(self, "class_to_idx") and self.class_to_idx:
            for k, v in self.class_to_idx.items():
                if "fake" in k.lower() or "ai" in k.lower() or "synth" in k.lower():
                    fake_idx = v
                    break

        with torch.no_grad():
            outputs = self.forward(x_rgb, return_breakdown=True)
            
            scaled_fusion_logits = torch.clamp(outputs["logits"], -12.0, 12.0) / temperature
            scaled_spatial_logits = torch.clamp(outputs["spatial_logits"], -12.0, 12.0) / temperature
            scaled_spectral_logits = torch.clamp(outputs["spectral_logits"], -12.0, 12.0) / temperature
            
            fusion_probs = F.softmax(scaled_fusion_logits, dim=1)
            spatial_probs = F.softmax(scaled_spatial_logits, dim=1)
            spectral_probs = F.softmax(scaled_spectral_logits, dim=1)
            
            ai_prob = fusion_probs[:, fake_idx].tolist()
            spatial_ai_prob = spatial_probs[:, fake_idx].tolist()
            spectral_ai_prob = spectral_probs[:, fake_idx].tolist()
            
            results = []
            for i in range(len(ai_prob)):
                prob = ai_prob[i]
                label = "AI-Generated" if prob >= 0.50 else "Real"
                confidence = prob if prob >= 0.50 else (1.0 - prob)
                
                results.append({
                    "label": label,
                    "is_ai": prob >= 0.50,
                    "confidence_pct": round(confidence * 100, 1),
                    "ai_probability": round(prob, 4),
                    "spatial_ai_probability": round(spatial_ai_prob[i], 4),
                    "spectral_ai_probability": round(spectral_ai_prob[i], 4)
                })
                
            return results


class PretrainedDualStreamDetector(nn.Module):
    """
    Production Architecture: ResNet-50 Spatial + Pretrained ResNet-18 Spectral (2D FFT).
    Provides superior feature representation across both spatial textures and frequency harmonics.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        base_spat = models.resnet50(weights=None)
        self.spatial_encoder = nn.Sequential(*list(base_spat.children())[:-1])
        
        base_spec = models.resnet18(weights=None)
        self.spectral_encoder = nn.Sequential(*list(base_spec.children())[:-1])
        
        self.fusion_head = nn.Sequential(
            nn.Linear(2048 + 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.spatial_head = nn.Linear(2048, num_classes)
        self.spectral_head = nn.Linear(512, num_classes)
        
    @property
    def spatial_aux_head(self):
        return self.spatial_head

    @property
    def spectral_aux_head(self):
        return self.spectral_head

    def extract_fft_spectrum(self, x_rgb: torch.Tensor) -> torch.Tensor:
        gray = 0.2989 * x_rgb[:, 0:1] + 0.5870 * x_rgb[:, 1:2] + 0.1140 * x_rgb[:, 2:3]
        fft = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft)
        mag = torch.log1p(torch.abs(fft_shifted))
        b = mag.shape[0]
        min_v = mag.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        max_v = mag.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        norm_mag = (mag - min_v) / (max_v - min_v + 1e-8)
        return (norm_mag - 0.5) / 0.5

    def forward(self, x: torch.Tensor, return_breakdown: bool = False):
        f_spat = self.spatial_encoder(x).flatten(1)
        x_fft = self.extract_fft_spectrum(x)
        f_spec = self.spectral_encoder(x_fft.repeat(1, 3, 1, 1)).flatten(1)
        
        fused = torch.cat([f_spat, f_spec], dim=1)
        logits = self.fusion_head(fused)
        
        if return_breakdown:
            return {
                'logits': logits,
                'spatial_logits': self.spatial_head(f_spat),
                'spectral_logits': self.spectral_head(f_spec)
            }
        return logits

    def predict_proba(self, x_rgb: torch.Tensor, temperature: float = 1.5):
        self.eval()
        fake_idx = 0
        if hasattr(self, "class_to_idx") and self.class_to_idx:
            for k, v in self.class_to_idx.items():
                if "fake" in k.lower() or "ai" in k.lower() or "synth" in k.lower():
                    fake_idx = v
                    break

        with torch.no_grad():
            outputs = self.forward(x_rgb, return_breakdown=True)
            
            scaled_fusion_logits = torch.clamp(outputs["logits"], -12.0, 12.0) / temperature
            scaled_spatial_logits = torch.clamp(outputs["spatial_logits"], -12.0, 12.0) / temperature
            scaled_spectral_logits = torch.clamp(outputs["spectral_logits"], -12.0, 12.0) / temperature
            
            fusion_probs = F.softmax(scaled_fusion_logits, dim=1)
            spatial_probs = F.softmax(scaled_spatial_logits, dim=1)
            spectral_probs = F.softmax(scaled_spectral_logits, dim=1)
            
            ai_prob = fusion_probs[:, fake_idx].tolist()
            spatial_ai_prob = spatial_probs[:, fake_idx].tolist()
            spectral_ai_prob = spectral_probs[:, fake_idx].tolist()
            
            results = []
            for i in range(len(ai_prob)):
                prob = ai_prob[i]
                label = "AI-Generated" if prob >= 0.50 else "Real"
                confidence = prob if prob >= 0.50 else (1.0 - prob)
                
                results.append({
                    "label": label,
                    "is_ai": prob >= 0.50,
                    "confidence_pct": round(confidence * 100, 1),
                    "ai_probability": round(prob, 4),
                    "spatial_ai_probability": round(spatial_ai_prob[i], 4),
                    "spectral_ai_probability": round(spectral_ai_prob[i], 4)
                })
                
            return results


class RadialProfileExtractor(nn.Module):
    """Computes 1D Azimuthal (radially averaged) power spectrum on GPU."""
    def __init__(self, size: int = 224):
        super().__init__()
        h, w = size, size
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        r = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2).long()
        self.max_r = min(cx, cy)
        r = torch.clamp(r, 0, self.max_r - 1)
        self.register_buffer('r_grid', r.view(-1))
        counts = torch.bincount(self.r_grid, minlength=self.max_r).float().clamp(min=1.0)
        self.register_buffer('counts', counts)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x_rgb.device).view(1, 3, 1, 1)
        gray = torch.sum(x_rgb * weights, dim=1, keepdim=True)
        fft2 = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft2, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft_shift)).squeeze(1)
        B, H, W = mag.shape
        mag_flat = mag.view(B, -1)
        rad_profiles = torch.zeros(B, self.max_r, device=x_rgb.device)
        rad_profiles.scatter_add_(1, self.r_grid.unsqueeze(0).expand(B, -1), mag_flat)
        rad_profiles = rad_profiles / self.counts.unsqueeze(0)
        p_min = rad_profiles.min(dim=1, keepdim=True)[0]
        p_max = rad_profiles.max(dim=1, keepdim=True)[0]
        return (rad_profiles - p_min) / (p_max - p_min + 1e-8)


class RadialDualStreamDetector(nn.Module):
    """
    Production Architecture: Spatial (ResNet-50) + Azimuthal Radial Spectral MLP Fusion.
    Captures 1/f^2 energy deviations across spatial frequencies (Durall et al., 2020).
    """
    def __init__(self, in_features: int = 112, num_classes: int = 2):
        super().__init__()
        base_spat = models.resnet50(weights=None)
        self.spatial_encoder = nn.Sequential(*list(base_spat.children())[:-1])
        
        self.radial_extractor = RadialProfileExtractor(size=224)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.spatial_head = nn.Linear(2048, num_classes)
        self.spectral_head = nn.Linear(64, num_classes)

    @property
    def spatial_aux_head(self):
        return self.spatial_head

    @property
    def spectral_aux_head(self):
        return self.spectral_head

    def forward(self, x: torch.Tensor, return_breakdown: bool = False):
        f_spat = self.spatial_encoder(x).flatten(1)
        rad_prof = self.radial_extractor(x)
        f_spec = self.spectral_mlp(rad_prof)
        
        fused = torch.cat([f_spat, f_spec], dim=1)
        logits = self.fusion_head(fused)
        if return_breakdown:
            return {
                'logits': logits,
                'spatial_logits': self.spatial_head(f_spat),
                'spectral_logits': self.spectral_head(f_spec)
            }
        return logits

    def predict_proba(self, x_rgb: torch.Tensor, temperature: float = 1.5):
        self.eval()
        fake_idx = 0
        if hasattr(self, "class_to_idx") and self.class_to_idx:
            for k, v in self.class_to_idx.items():
                if "fake" in k.lower() or "ai" in k.lower() or "synth" in k.lower():
                    fake_idx = v
                    break

        with torch.no_grad():
            outputs = self.forward(x_rgb, return_breakdown=True)
            scaled_fusion_logits = torch.clamp(outputs["logits"], -12.0, 12.0) / temperature
            scaled_spatial_logits = torch.clamp(outputs["spatial_logits"], -12.0, 12.0) / temperature
            scaled_spectral_logits = torch.clamp(outputs["spectral_logits"], -12.0, 12.0) / temperature
            
            fusion_probs = F.softmax(scaled_fusion_logits, dim=1)
            spatial_probs = F.softmax(scaled_spatial_logits, dim=1)
            spectral_probs = F.softmax(scaled_spectral_logits, dim=1)
            
            ai_prob = fusion_probs[:, fake_idx].tolist()
            spatial_ai_prob = spatial_probs[:, fake_idx].tolist()
            spectral_ai_prob = spectral_probs[:, fake_idx].tolist()
            
            results = []
            for i in range(len(ai_prob)):
                prob = ai_prob[i]
                label = "AI-Generated" if prob >= 0.50 else "Real"
                confidence = prob if prob >= 0.50 else (1.0 - prob)
                results.append({
                    "label": label,
                    "is_ai": prob >= 0.50,
                    "confidence_pct": round(confidence * 100, 1),
                    "ai_probability": round(prob, 4),
                    "spatial_ai_probability": round(spatial_ai_prob[i], 4),
                    "spectral_ai_probability": round(spectral_ai_prob[i], 4)
                })
            return results


class SpatialStreamDetector(nn.Module):
    """Spatial Stream backbone: ResNet-50 fine-tuned for facial deepfake textures."""
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.classifier(feats)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Support 2-stage MLP classifier head from unified multi-source training
        if "classifier.4.weight" in state_dict and len(self.classifier) < 5:
            device = next(self.parameters()).device
            self.classifier = nn.Sequential(
                nn.Linear(2048, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 2)
            ).to(device)

        clean_sd = {}
        for k, v in state_dict.items():
            if k == "classifier.weight" and "classifier.1.weight" in self.state_dict():
                clean_sd["classifier.1.weight"] = v
            elif k == "classifier.bias" and "classifier.1.bias" in self.state_dict():
                clean_sd["classifier.1.bias"] = v
            elif k == "classifier.1.weight" and "classifier.weight" in self.state_dict():
                clean_sd["classifier.weight"] = v
            elif k == "classifier.1.bias" and "classifier.bias" in self.state_dict():
                clean_sd["classifier.bias"] = v
            else:
                clean_sd[k] = v
        return super().load_state_dict(clean_sd, strict=strict)


class SpectralStreamDetector(nn.Module):
    """Spectral Stream backbone: ResNet-18 fine-tuned on 2D FFT magnitude spectrum."""
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.classifier = nn.Linear(in_features, num_classes)

    def extract_fft_spectrum(self, x_rgb: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x_rgb.device).view(1, 3, 1, 1)
        gray = torch.sum(x_rgb * weights, dim=1, keepdim=True)
        fft2 = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft_shifted))
        b = mag.shape[0]
        mean = mag.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        std = mag.view(b, -1).std(dim=1).view(b, 1, 1, 1).clamp(min=1e-6)
        norm_mag = (mag - mean) / std
        return norm_mag.repeat(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = self.extract_fft_spectrum(x)
        feats = self.encoder(x_fft)
        return self.classifier(feats)


class DedicatedSpectralDetector(nn.Module):
    """
    Dedicated 4-Layer Frequency CNN (405k Parameters) specifically designed
    for 2D FFT log-magnitude maps: prevents ImageNet-overfitting on frequency space.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: Fine high-frequency grid lines (224 -> 112)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            
            # Stage 2: Spliced boundary harmonics (112 -> 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            
            # Stage 3: Mid-frequency spectral rings (56 -> 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            
            # Stage 4: Global spectral energy distribution (28 -> 14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 Quadrant Pooling preserves low/mid/high frequency bands
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def extract_fft(self, x_rgb: torch.Tensor) -> torch.Tensor:
        inv_mean = torch.tensor([0.485, 0.456, 0.406], device=x_rgb.device).view(1, 3, 1, 1)
        inv_std = torch.tensor([0.229, 0.224, 0.225], device=x_rgb.device).view(1, 3, 1, 1)
        rgb_raw = torch.clamp(x_rgb * inv_std + inv_mean, 0.0, 1.0)
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb_raw.device).view(1, 3, 1, 1)
        gray = torch.sum(rgb_raw * weights, dim=1, keepdim=True)
        fft2 = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft_shifted))
        b = mag.shape[0]
        min_v = mag.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        max_v = mag.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        return (mag - min_v) / (max_v - min_v + 1e-8)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        x_fft = self.extract_fft(x_rgb)
        feats = self.features(x_fft)
        return self.classifier(feats)


class AUCWeightedDualStreamDetector(nn.Module):
    """
    VeriSight Production Multi-Domain Model:
    Combines Spatial Stream (ResNet-50) and Spectral Stream (Dedicated 4-Layer CNN / ResNet-18)
    via empirical decision fusion: P_dual = w_spatial * P_spatial + w_spectral * P_spectral
    """
    def __init__(self, w_spatial: float = 0.5, w_spectral: float = 0.5, num_classes: int = 2, spectral_type: str = "dedicated"):
        super().__init__()
        self.w_spatial = float(w_spatial)
        self.w_spectral = float(w_spectral)
        self.spatial_model = SpatialStreamDetector(num_classes=num_classes)
        if spectral_type == "dedicated":
            self.spectral_model = DedicatedSpectralDetector(num_classes=num_classes)
        else:
            self.spectral_model = SpectralStreamDetector(num_classes=num_classes)
        self.class_to_idx = {"FAKE": 0, "REAL": 1}

    def forward(self, x: torch.Tensor, return_breakdown: bool = False):
        spat_logits = self.spatial_model(x)
        spec_logits = self.spectral_model(x)
        
        spat_probs = F.softmax(spat_logits, dim=1)
        spec_probs = F.softmax(spec_logits, dim=1)
        dual_probs = self.w_spatial * spat_probs + self.w_spectral * spec_probs
        dual_logits = torch.log(dual_probs.clamp(min=1e-7))
        
        if return_breakdown:
            return {
                'logits': dual_logits,
                'spatial_logits': spat_logits,
                'spectral_logits': spec_logits
            }
        return dual_logits

    def predict_proba(self, x_rgb: torch.Tensor, temperature: float = 1.0):
        self.eval()
        fake_idx = 0
        if hasattr(self, "class_to_idx") and self.class_to_idx:
            for k, v in self.class_to_idx.items():
                if "fake" in k.lower() or "ai" in k.lower():
                    fake_idx = v
                    break

        with torch.no_grad():
            outputs = self.forward(x_rgb, return_breakdown=True)
            spat_probs = F.softmax(outputs["spatial_logits"], dim=1)
            spec_probs = F.softmax(outputs["spectral_logits"], dim=1)
            dual_probs = self.w_spatial * spat_probs + self.w_spectral * spec_probs
            
            ai_prob = dual_probs[:, fake_idx].tolist()
            spatial_ai_prob = spat_probs[:, fake_idx].tolist()
            spectral_ai_prob = spec_probs[:, fake_idx].tolist()
            
            results = []
            for i in range(len(ai_prob)):
                prob = ai_prob[i]
                label = "AI-Generated" if prob >= 0.50 else "Real"
                confidence = prob if prob >= 0.50 else (1.0 - prob)
                results.append({
                    "label": label,
                    "is_ai": prob >= 0.50,
                    "confidence_pct": round(confidence * 100, 1),
                    "ai_probability": round(prob, 4),
                    "spatial_ai_probability": round(spatial_ai_prob[i], 4),
                    "spectral_ai_probability": round(spectral_ai_prob[i], 4)
                })
            return results




