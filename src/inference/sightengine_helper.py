"""
VeriSight - Auxiliary Commercial API Verification Helper
Optional local module for querying Sightengine API as an external cross-validation baseline.

CONFIDENTIAL / INTERNAL USE ONLY:
This module is strictly for local verification / testing and MUST NEVER be referenced
in academic reports, viva presentations, or submitted project documentation.
"""

import os
from pathlib import Path
import requests

# Load optional .env from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

if DOTENV_PATH.exists():
    try:
        with open(DOTENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

SIGHTENGINE_API_USER = os.environ.get("SIGHTENGINE_API_USER", "").strip()
SIGHTENGINE_API_SECRET = os.environ.get("SIGHTENGINE_API_SECRET", "").strip()
ENABLE_SIGHTENGINE = os.environ.get("ENABLE_SIGHTENGINE", "false").lower() in ("true", "1")

SIGHTENGINE_IMAGE_ENDPOINT = "https://api.sightengine.com/1.0/check.json"


def is_sightengine_available() -> bool:
    """Returns True if Sightengine credentials are configured and feature is enabled."""
    return bool(SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET and ENABLE_SIGHTENGINE)


def check_image_sightengine(image_bytes: bytes, filename: str = "image.jpg") -> dict | None:
    """
    Sends an image to Sightengine to check for AI-generation and deepfake manipulation.
    Returns normalized score dict or None if unavailable / failed.
    """
    if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET):
        return None

    data = {
        "api_user": SIGHTENGINE_API_USER,
        "api_secret": SIGHTENGINE_API_SECRET,
        "models": "genai,deepfake",
    }
    files = {
        "media": (filename, image_bytes, "image/jpeg")
    }

    try:
        response = requests.post(
            SIGHTENGINE_IMAGE_ENDPOINT,
            data=data,
            files=files,
            timeout=(10, 30)
        )
        if response.status_code != 200:
            return None

        result = response.json()
        if result.get("status") != "success":
            return None

        # Extract genai score
        ai_score = result.get("type", {}).get("ai_generated")
        if ai_score is not None:
            ai_score = float(ai_score)

        # Extract deepfake score if faces were analyzed
        deepfake_score = None
        faces = result.get("faces", [])
        if faces and isinstance(faces, list):
            df_scores = [f.get("deepfake", 0.0) for f in faces if "deepfake" in f]
            if df_scores:
                deepfake_score = max(df_scores)

        return {
            "ai_generated_score": ai_score,
            "deepfake_score": deepfake_score,
            "raw": result
        }
    except Exception:
        return None
