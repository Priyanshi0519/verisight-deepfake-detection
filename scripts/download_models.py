#!/usr/bin/env python3
"""
CLI utility to download and verify VeriSight pre-trained model weights.
Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_downloader import ensure_models_available, MODELS_CONFIG

def main():
    print("=" * 70)
    print(" VeriSight - Pre-trained Model Checkpoint Downloader")
    print("=" * 70)
    print("Checking model storage directory: saved_models/")
    
    target_dir = PROJECT_ROOT / "saved_models"
    success = ensure_models_available(target_dir=target_dir, auto_download=True)
    
    print("-" * 70)
    if success:
        print("All model weights are verified and ready for inference!")
    else:
        print("Some model checkpoints are missing.")
        print("You can start the inference server in initialized test mode, or")
        print("upload weights to Google Drive and configure file IDs in model_downloader.py.")
    print("=" * 70)

if __name__ == "__main__":
    main()
