# VeriSight: Dual-Stream Spatial and Spectral Deepfake & AI Image Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Chrome Extension: MV3](https://img.shields.io/badge/Extension-Manifest%20V3-yellow.svg)](extension/)

**VeriSight** is an end-to-end computer vision and digital forensics system engineered to detect **deepfake facial manipulations** (face swaps, facial reenactment) and **full-frame generative AI imagery** (Diffusion models, GANs). 

It pairs a **Dual-Stream Neural Network** combining Spatial RGB residual features with 2D Fast Fourier Transform (2D FFT) spectral power distributions, and serves predictions in real time through a lightweight **Flask REST API**, a standalone **interactive web demonstration bench**, and a viewport-aware **Google Chrome Extension (Manifest V3)** for live web browsing protection.

---

## Key Features

1. **Dual-Stream Spatial + Spectral Architecture**:
   - **Spatial Stream**: Extracts deep pixel texture, blending boundaries, and facial landmark inconsistencies using a ResNet-based backbone.
   - **Spectral Stream**: Converts images to the frequency domain via 2D Fast Fourier Transform (`2D FFT`) and radial power distributions. Generative upsampling layers (transposed convolutions) introduce characteristic periodic high-frequency grid artifacts that are clearly separable in the Fourier domain.
   - **AUC-Calibrated Fusion**: Employs dynamic confidence weighting between spatial facial tamper detection and global spectral anomalies.

2. **Three-Class Forensic Classification**:
   - Real Camera Photography: Authentic unmanipulated documentary photos.
   - Deepfake (Face Manipulation): Target face swapped or expression-manipulated (FaceForensics++, DeepfakeDetection).
   - Full AI-Generated Synthetic: Entire image synthesized by generative AI (Stable Diffusion, Midjourney, DALL-E, Flux, GANs).

3. **Viewport-Aware Chrome Extension (Manifest V3)**:
   - Uses `IntersectionObserver` with a 500ms debounce dwell buffer to scan only images the user actually pauses on.
   - Non-destructive floating badges directly over web images with click-to-expand forensic detail cards.
   - Client-side LRU hashing to prevent redundant network inference.

4. **Interactive Web Demonstration Bench (`demo.html`)**:
   - Live drag-and-drop image upload and direct URL inspection.
   - Built-in 8-sample evaluation gallery spanning all 3 classes.
   - One-click automated benchmark runner with live scorecard and latency metrics.

---

## Repository Structure

```text
verisight-deepfake-detection/
├── demo.html               # Standalone interactive 3-class web demo
├── requirements.txt        # Python package dependencies
├── LICENSE                 # MIT License
├── README.md               # Project documentation
├── extension/              # Chrome Extension (Manifest V3)
│   ├── manifest.json       # MV3 configuration & permissions
│   ├── background.js       # Service worker & batch request router
│   ├── content.js          # Viewport observer & badge injection
│   ├── content.css         # Badge & forensic modal styling
│   ├── popup.html          # Extension control panel
│   ├── popup.js            # Toggle auto-scan & server health check
│   └── icons/              # Extension status icons
├── notebooks/              # Jupyter research & training notebooks
│   ├── 01_dataset_download_and_inspection.ipynb
│   ├── 03_dual_stream_spectral_detection.ipynb
│   ├── 05_faceforensics_spectral_training.ipynb
│   ├── 08_spatial_vs_spectral_cross_validation.ipynb
│   ├── 11_train_unified_multi_source_dataset.ipynb
│   └── 13_train_faceforensics_clean_deepfake_detector.ipynb
├── saved_models/           # Target directory for .pth model weights
│   ├── .gitkeep
│   └── README.md
├── scripts/                # Utility scripts
│   ├── download_models.py  # Automated checkpoint downloader from Google Drive
│   └── demo_spectral.py    # 6-panel spatial vs 2D FFT comparison tool
└── src/                    # Core source code
    ├── inference/
    │   ├── app.py          # Flask REST API server (endpoints: /predict, /predict-batch)
    │   └── sightengine_helper.py
    ├── preprocessing/
    │   └── spectral.py     # 2D FFT magnitude & 1D azimuthal power extraction
    ├── training/
    │   ├── dual_stream_model.py # Spatial + Spectral PyTorch architectures
    │   └── train.py        # Model training loop with AUC metric tracking
    └── utils/
        └── model_downloader.py # Google Drive download & local cache manager
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Google Chrome (for the browser extension)

### 1. Clone the Repository
```bash
git clone https://github.com/Priyanshi0519/verisight-deepfake-detection.git
cd verisight-deepfake-detection
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Pre-trained Model Weights

Due to GitHub's 100MB file limit, trained PyTorch weights are stored externally on Google Drive.

### Automatic Download
When you run the inference backend, `src/utils/model_downloader.py` will automatically verify and download the checkpoints:
```bash
python src/inference/app.py
```

### Manual Download
You can also download the weights ahead of time using:
```bash
python scripts/download_models.py
```

### Checkpoints:
| Model File | Description | Size | Validation Acc |
| :--- | :--- | :--- | :--- |
| `verisight_unified_face_production.pth` | Unified Multi-Source Face Model | ~102 MB | **98.3%** |
| `face_deepfake_stream.pth` | FaceForensics++ Tampering Stream | ~98 MB | **92.4%** |
| `ai_generative_stream.pth` | Dual-Stream Spatial + 2D FFT AI Model | ~98 MB | **94.8%** |

*(If hosting your own weights on Google Drive, simply set `VERISIGHT_FACE_DRIVE_ID`, `VERISIGHT_TAMPER_DRIVE_ID`, and `VERISIGHT_AI_DRIVE_ID` in your environment or update `src/utils/model_downloader.py`.)*

---

## Running the Application

### 1. Start the Inference Server
```bash
python src/inference/app.py
```
Output:
```text
Model directory: saved_models
Loading weights from: verisight_unified_face_production.pth
  Loaded DualStreamDetector (Spatial + 2D FFT Spectral CNN, Acc=98.3%).
Face detection: MediaPipe Tasks (deep learning, confidence >= 0.60)
Starting VeriSight API on http://127.0.0.1:5000
```

### 2. Open the Web Benchmark
Once the server is running, navigate to:
- **Local Browser**: `http://127.0.0.1:5000/demo` (or double-click `demo.html`)

You can drag & drop any image, test external URLs, or click **"Run Full 8-Sample Suite"** to verify system accuracy across all 3 classes.

### 3. Install the Chrome Extension
1. Open Google Chrome and go to `chrome://extensions`.
2. Enable **Developer mode** (top-right toggle).
3. Click **Load unpacked** (top-left button).
4. Select the `extension/` directory inside this repository.
5. The VeriSight shield icon will appear in your Chrome toolbar. Open any website (e.g. Google Images or social media) and hover/pause over images to see live forensic verification.

---

## Academic Information

- **Project**: B.Tech Final Year Capstone Project
- **Institution**: Motilal Nehru National Institute of Technology (MNNIT) Allahabad
- **Department**: Department of Computer Science and Engineering
- **Author**: Priyanshi

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
