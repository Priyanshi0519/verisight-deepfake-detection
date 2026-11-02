"""
VeriSight - Pre-trained Model Downloader & Manager
Handles automatic downloading of neural network checkpoints from Google Drive / Cloud storage,
with local disk caching and fallback support.
"""

import os
import sys
import shutil
import urllib.request
import urllib.parse
from pathlib import Path

# Google Drive File IDs for trained checkpoints
# Replace with your public Google Drive file IDs or set corresponding environment variables
MODELS_CONFIG = {
    "verisight_unified_face_production.pth": {
        "drive_id": os.environ.get("VERISIGHT_FACE_DRIVE_ID", ""),
        "direct_url": os.environ.get("VERISIGHT_FACE_URL", ""),
        "description": "Unified Multi-Source Face Forensics Model (98.3% accuracy)",
        "approx_size_mb": 102
    },
    "face_deepfake_stream.pth": {
        "drive_id": os.environ.get("VERISIGHT_TAMPER_DRIVE_ID", ""),
        "direct_url": os.environ.get("VERISIGHT_TAMPER_URL", ""),
        "description": "FaceForensics++ Dual-Stream Manipulation Specialist",
        "approx_size_mb": 98
    },
    "ai_generative_stream.pth": {
        "drive_id": os.environ.get("VERISIGHT_AI_DRIVE_ID", ""),
        "direct_url": os.environ.get("VERISIGHT_AI_URL", ""),
        "description": "Dual-Stream Spatial + 2D FFT Spectral Generative Detector",
        "approx_size_mb": 98
    }
}

LOCAL_FALLBACK_DIRS = [
    Path("saved_models"),
    Path("../deepfake-detection/saved_models"),
    Path(r"C:\VeriSight\models"),
]


def download_from_google_drive(file_id: str, destination: Path) -> bool:
    """
    Downloads a large file from Google Drive handling the virus-scan confirmation token.
    """
    try:
        import gdown
        print(f"Downloading from Google Drive via gdown: {destination.name}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        return destination.exists() and destination.stat().st_size > 1024 * 1024
    except ImportError:
        pass

    print(f"Downloading from Google Drive (standard client): {destination.name}...")
    base_url = "https://docs.google.com/uc?export=download"
    session_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    
    req = urllib.request.Request(
        f"{base_url}&id={file_id}",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    try:
        with session_opener.open(req) as response:
            content = response.read(65536)
            text_preview = content.decode("utf-8", errors="ignore")
            
            # Check for Google Drive confirmation token
            confirm_token = None
            if "confirm=" in text_preview:
                import re
                match = re.search(r"confirm=([0-9A-Za-z_]+)", text_preview)
                if match:
                    confirm_token = match.group(1)
            
            if confirm_token:
                download_url = f"{base_url}&id={file_id}&confirm={confirm_token}"
            else:
                download_url = f"{base_url}&id={file_id}"

        # Download with chunked progress
        with session_opener.open(urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})) as src:
            with open(destination, "wb") as dst:
                chunk_size = 1024 * 1024
                downloaded = 0
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    downloaded += len(chunk)
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Progress: {mb:.1f} MB downloaded...", end="", flush=True)
                print()

        return destination.exists() and destination.stat().st_size > 1024 * 1024
    except Exception as e:
        print(f"\nDownload failed for {destination.name}: {e}")
        if destination.exists():
            destination.unlink()
        return False


def ensure_models_available(target_dir: Path = None, auto_download: bool = True) -> bool:
    """
    Checks if required model weights exist in target_dir.
    1. If missing, attempts to copy from local fallback directories.
    2. If not found locally, attempts to download from Google Drive or direct URL.
    """
    if target_dir is None:
        target_dir = Path("saved_models")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    all_ready = True
    
    for filename, config in MODELS_CONFIG.items():
        dest_path = target_dir / filename
        
        # 1. Check if already present
        if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
            continue
            
        # 2. Check local fallback directories
        found_local = False
        for fallback_dir in LOCAL_FALLBACK_DIRS:
            local_candidate = fallback_dir / filename
            if local_candidate.exists() and local_candidate.stat().st_size > 1024 * 1024:
                print(f"[Model Manager] Found local cached weights at: {local_candidate.resolve()}")
                try:
                    shutil.copy2(local_candidate, dest_path)
                    print(f"  -> Synchronized to {dest_path}")
                    found_local = True
                    break
                except Exception as e:
                    print(f"  -> Could not copy local cache: {e}")
        
        if found_local:
            continue
            
        # 3. Download from Google Drive / URL if enabled
        if auto_download:
            drive_id = config.get("drive_id", "").strip()
            direct_url = config.get("direct_url", "").strip()
            
            if drive_id:
                print(f"[Model Manager] Downloading {filename} from Google Drive...")
                success = download_from_google_drive(drive_id, dest_path)
                if success:
                    print(f"  -> Successfully installed {filename}")
                    continue
            elif direct_url:
                print(f"[Model Manager] Downloading {filename} from direct URL...")
                try:
                    urllib.request.urlretrieve(direct_url, dest_path)
                    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
                        print(f"  -> Successfully installed {filename}")
                        continue
                except Exception as e:
                    print(f"  -> Download failed: {e}")
        
        # If not found and not downloaded
        all_ready = False
        print(f"[Model Manager] Notice: {filename} ({config['description']}) is not present in {target_dir}.")
        print("  To provide weights:")
        print(f"  - Place your trained '{filename}' in {target_dir.resolve()}")
        print("  - Or add your shared Google Drive file ID in src/utils/model_downloader.py")
        
    return all_ready


if __name__ == "__main__":
    ensure_models_available(auto_download=True)
