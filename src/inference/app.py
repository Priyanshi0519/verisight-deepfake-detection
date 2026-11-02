"""
VeriSight: Multi-Factor Forensic Ensemble Service for Deepfake & AI Detection
------------------------------------------------------------------------------
v2.0 — Enhanced with CLIP zero-shot verification, MediaPipe face detection,
temperature-calibrated confidence, and 3-way weighted voting.

Expert Streams:
1. Face Forensics Stream (CIPLAB/FF++ ResNet-50 + 2D FFT):
   Detects facial deepfake manipulations (FaceSwap, Face2Face, NeuralTextures).
2. Generative AI Stream (CIFAKE ResNet-50 + 2D FFT):
   Detects full-frame AI diffusion synthesis (Stable Diffusion, Midjourney, DALL-E).
3. CLIP Zero-Shot Semantic Verifier:
   OpenAI CLIP trained on 400M web image-text pairs provides a semantic reality check
   that corrects domain-shift false positives from narrow CNN training distributions.
"""

import os
import sys
import io
import time
import base64
import tempfile
import urllib.request
from pathlib import Path
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.training.dual_stream_model import DualStreamDetector, ColabDualStreamDetector, PretrainedDualStreamDetector, RadialDualStreamDetector, AUCWeightedDualStreamDetector
except (ImportError, ModuleNotFoundError):
    from ..training.dual_stream_model import DualStreamDetector, ColabDualStreamDetector, PretrainedDualStreamDetector, RadialDualStreamDetector, AUCWeightedDualStreamDetector

try:
    from src.inference.sightengine_helper import check_image_sightengine, is_sightengine_available
except (ImportError, ModuleNotFoundError):
    try:
        from .sightengine_helper import check_image_sightengine, is_sightengine_available
    except Exception:
        check_image_sightengine = lambda *args, **kwargs: None
        is_sightengine_available = lambda: False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Operational Mode: Set USE_PURE_CNN=True to run 100% purely on Dual-Stream CNN
# (Spatial ResNet-50 + 2D FFT Spectral Stream), disabling CLIP foundation models.
USE_PURE_CNN = os.environ.get("USE_PURE_CNN", "false").lower() in ("true", "1")

# Model directory: configurable via env var, with local fallback to avoid OneDrive bloat
_DEFAULT_LOCAL_MODEL_DIR = Path(r"C:\VeriSight\models")
_PROJECT_MODEL_DIR = Path("saved_models")

if os.environ.get("VERISIGHT_MODELS_DIR"):
    MODEL_DIR = Path(os.environ["VERISIGHT_MODELS_DIR"])
elif _DEFAULT_LOCAL_MODEL_DIR.exists() and any(_DEFAULT_LOCAL_MODEL_DIR.glob("*.pth")):
    MODEL_DIR = _DEFAULT_LOCAL_MODEL_DIR
else:
    MODEL_DIR = _PROJECT_MODEL_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Auto-download models if missing
try:
    from src.utils.model_downloader import ensure_models_available
except ImportError:
    try:
        from utils.model_downloader import ensure_models_available
    except ImportError:
        ensure_models_available = None

print(f"Model directory: {MODEL_DIR.resolve()}")

# LRU Cache for previously scanned image URLs (max 1000 items)
class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity
        
    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

URL_CACHE = LRUCache(capacity=1000)

# Preprocessing transform for CNN models
PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global models
model_face = None
model_face_tamper = None
model_ai = None
clip_model = None
clip_preprocess = None
clip_tokenizer = None
is_trained_weights_loaded = False

# ── Face Detection (MediaPipe tasks API v1.0+ or Haar Cascade fallback) ──────
mp_face_detector = None
haar_cascade = None

def init_face_detector():
    global mp_face_detector, haar_cascade
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        import urllib.request as _urlreq
        import tempfile, os

        # Download the face detection model if not cached
        model_path = os.path.join(tempfile.gettempdir(), "blaze_face_short_range.tflite")
        if not os.path.exists(model_path):
            print("Downloading MediaPipe face detection model...")
            _urlreq.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
                model_path
            )

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            min_detection_confidence=0.60
        )
        mp_face_detector = vision.FaceDetector.create_from_options(options)
        print("Face detection: MediaPipe Tasks (deep learning, calibrated confidence >= 0.60)")
    except Exception as e:
        mp_face_detector = None
        print(f"MediaPipe unavailable ({e}), using Haar Cascade fallback")
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Face detection: OpenCV Haar Cascade")


def detect_faces_and_bboxes(image: Image.Image) -> tuple[bool, tuple | None]:
    """
    Detects valid human faces in an image with strict confidence and size filtering.
    Filters out background pareidolia (e.g. fantasy rocks, clouds, leaves).
    Returns (has_face: bool, primary_bbox: (x, y, w, h) or None).
    """
    arr = np.array(image)
    img_w, img_h = image.size
    
    if mp_face_detector is not None:
        import mediapipe as mp
        if arr.ndim == 2:
            rgb = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            rgb = arr[:, :, :3]
        else:
            rgb = arr.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = mp_face_detector.detect(mp_image)
        
        valid_boxes = []
        for det in result.detections:
            score = det.categories[0].score
            bb = det.bounding_box
            w, h = bb.width, bb.height
            area_ratio = (w * h) / float(img_w * img_h)
            # Require minimum detection confidence 0.60, minimum size 45x45, and area >= 0.5%
            if score >= 0.60 and w >= 45 and h >= 45 and area_ratio >= 0.005:
                valid_boxes.append((bb.origin_x, bb.origin_y, w, h, score))
                
        if valid_boxes:
            valid_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            return True, valid_boxes[0][:4]
        return False, None
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        cascade = haar_cascade or cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))
        valid_boxes = []
        for (x, y, w, h) in faces:
            area_ratio = (w * h) / float(img_w * img_h)
            if area_ratio >= 0.005:
                valid_boxes.append((x, y, w, h))
        if valid_boxes:
            valid_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            return True, valid_boxes[0]
        return False, None


def detect_faces(image_array: np.ndarray) -> bool:
    has_face, _ = detect_faces_and_bboxes(Image.fromarray(image_array))
    return has_face


# ── CLIP Zero-Shot Semantic Verifier ─────────────────────────────────────────
CLIP_PROMPTS_AI = [
    "an AI generated synthetic image",
    "a computer generated digital picture made by generative AI",
    "an image created by Midjourney or Stable Diffusion or DALL-E or Flux",
    "a digital fantasy illustration, CGI render, or AI artwork",
    "photorealistic synthetic picture generated by AI",
]
CLIP_PROMPTS_REAL = [
    "a real photograph taken by a physical camera",
    "an authentic documentary camera photograph captured in the real world",
    "a real photo taken by a digital camera or smartphone",
    "an authentic unedited photo of a real scene or subject",
    "a genuine photograph with natural camera lens depth and sensor noise",
]


def init_clip_model():
    global clip_model, clip_preprocess, clip_tokenizer
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        clip_model = model
        clip_preprocess = preprocess
        clip_tokenizer = tokenizer
        print("CLIP verifier: ViT-B/32 (LAION-2B, 400M image-text pairs)")
    except Exception as e:
        clip_model = None
        print(f"CLIP verifier: disabled ({e})")


def clip_ai_probability(image: Image.Image) -> float:
    if clip_model is None:
        return 0.5
    
    img_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    
    all_prompts = CLIP_PROMPTS_AI + CLIP_PROMPTS_REAL
    tokens = clip_tokenizer(all_prompts).to(DEVICE)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)
        text_features = clip_model.encode_text(tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarities = (image_features @ text_features.T).squeeze(0)
        
        n_ai = len(CLIP_PROMPTS_AI)
        ai_score = similarities[:n_ai].mean().item()
        real_score = similarities[n_ai:].mean().item()
        
        logits = torch.tensor([ai_score, real_score]) / 0.05
        probs = F.softmax(logits, dim=0)
        
        return probs[0].item()


# ── CNN Model Loader ────────────────────────────────────────────────────────

def _load_single_model(path: Path):
    if path is None or not path.exists():
        return None, 0.0
        
    print(f"Loading weights from: {path.name}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    
    # Check if checkpoint is AUC-Weighted Dual-Stream (ResNet-50 Spatial + 2D FFT ResNet-18 Spectral)
    if isinstance(checkpoint, dict) and "spatial_state_dict" in checkpoint and "spectral_state_dict" in checkpoint:
        w_spat = checkpoint.get("w_spatial", 0.5)
        w_spec = checkpoint.get("w_spectral", 0.5)
        m = AUCWeightedDualStreamDetector(w_spatial=w_spat, w_spectral=w_spec, num_classes=2).to(DEVICE)
        m.spatial_model.load_state_dict(checkpoint["spatial_state_dict"])
        m.spectral_model.load_state_dict(checkpoint["spectral_state_dict"])
        m.class_to_idx = checkpoint.get("class_to_idx", {"FAKE": 0, "REAL": 1})
        m.eval()
        val_acc = checkpoint.get("val_accuracy", 0.0)
        print(f"  Loaded AUCWeightedDualStreamDetector (Spatial w={w_spat:.2f}, Spectral w={w_spec:.2f}, Acc={val_acc*100:.1f}%).")
        return m, val_acc

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Check if checkpoint is Radial Dual-Stream (ResNet-50 Spatial + 1D Radial Spectral MLP, in_dim=2112)
    if "fusion_head.0.weight" in state_dict and state_dict["fusion_head.0.weight"].shape[1] == 2112:
        m = RadialDualStreamDetector(num_classes=2).to(DEVICE)
        m.load_state_dict(state_dict, strict=True)
        print(f"  Loaded RadialDualStreamDetector strictly (ResNet-50 Spatial + 1D Radial Spectral MLP).")
    # Check if checkpoint is Pretrained Dual-Stream (ResNet-50 Spatial + ResNet-18 Spectral, in_dim=2560)
    elif "fusion_head.0.weight" in state_dict and state_dict["fusion_head.0.weight"].shape[1] == 2560:
        m = PretrainedDualStreamDetector(num_classes=2).to(DEVICE)
        m.load_state_dict(state_dict, strict=True)
        print(f"  Loaded PretrainedDualStreamDetector strictly (ResNet-50 Spatial + ResNet-18 Spectral).")
    # Check if checkpoint was trained with Colab sequential architecture (e.g. ai_generative_stream_best.pth)
    elif "spatial_encoder.0.weight" in state_dict and "fusion_head.0.weight" in state_dict and state_dict["fusion_head.0.weight"].shape[1] == 2304:
        m = ColabDualStreamDetector(num_classes=2).to(DEVICE)
        m.load_state_dict(state_dict, strict=True)
        print(f"  Loaded ColabDualStreamDetector strictly (100% parameter match: 361/361 keys).")
    else:
        backbone = "resnet18"
        if "fusion_head.0.weight" in state_dict:
            in_dim = state_dict["fusion_head.0.weight"].shape[1]
            if in_dim == 2304:
                backbone = "resnet50"
            elif in_dim == 768:
                backbone = "resnet18"
                
        m = DualStreamDetector(backbone=backbone, pretrained=False).to(DEVICE)
        clean_sd = {}
        for k, v in state_dict.items():
            clean_k = k.replace("spatial_aux_head.", "spatial_head.").replace("spectral_aux_head.", "spectral_head.")
            clean_sd[clean_k] = v
        m.load_state_dict(clean_sd, strict=False)
        print(f"  Loaded DualStreamDetector ({backbone}).")
        
    m.class_to_idx = checkpoint.get("class_to_idx", {"FAKE": 0, "REAL": 1})
    m.eval()
    val_acc = checkpoint.get("val_accuracy", 0.0)
    return m, val_acc


def init_ensemble_detectors():
    global model_face, model_face_tamper, model_ai, is_trained_weights_loaded
    
    # Check and download models from Google Drive / local cache if needed
    if ensure_models_available:
        ensure_models_available(MODEL_DIR, auto_download=True)
    
    init_face_detector()
    if not USE_PURE_CNN:
        init_clip_model()
    else:
        print("Pure-CNN Production Mode Active: CLIP disabled. Relying 100% on Dual-Stream Spatial + 2D FFT Spectral CNN.")
    
    # Load primary face forensics model (prioritizing 98.3% Unified Multi-Source Model)
    face_path = None
    for candidate in ["verisight_unified_face_production.pth", "verisight_face_production.pth", "face_deepfake_stream.pth", "dual_stream_faceforensics_best.pth"]:
        if (MODEL_DIR / candidate).exists():
            face_path = MODEL_DIR / candidate
            break
            
    # Load auxiliary facial tampering stream (FaceForensics++ specialist)
    tamper_path = None
    for candidate in ["face_deepfake_stream.pth", "dual_stream_faceforensics_best.pth"]:
        if (MODEL_DIR / candidate).exists() and (MODEL_DIR / candidate) != face_path:
            tamper_path = MODEL_DIR / candidate
            break
    
    # Load AI generative model (explicit candidates only)
    ai_path = None
    for candidate in ["verisight_dual_stream_production.pth", "ai_generative_stream_best.pth", "ai_generative_stream.pth", "cifake_generative_stream.pth"]:
        if (MODEL_DIR / candidate).exists():
            ai_path = MODEL_DIR / candidate
            break
    
    if face_path is None:
        print(f"WARNING: No face forensics model found in {MODEL_DIR}")
    if ai_path is None:
        print(f"WARNING: No AI generative model found in {MODEL_DIR}")
        
    model_face, acc_face = _load_single_model(face_path)
    if tamper_path is not None:
        model_face_tamper, _ = _load_single_model(tamper_path)
    else:
        model_face_tamper = None
        
    model_ai, acc_ai = _load_single_model(ai_path)
    
    if model_face is not None and model_ai is not None:
        is_trained_weights_loaded = True
        print(f"Ensemble Online: Face Model ({acc_face*100:.1f}%), AI Model ({acc_ai*100:.1f}%)")
        print(f"  Face class_to_idx: {getattr(model_face, 'class_to_idx', 'N/A')}")
        print(f"  AI    class_to_idx: {getattr(model_ai, 'class_to_idx', 'N/A')}")
    else:
        is_trained_weights_loaded = False
        print("ERROR: Cannot start ensemble — missing model checkpoints.")


init_ensemble_detectors()


# ── Prediction Engine with 3-Way Voting ─────────────────────────────────────

TEMPERATURE = 1.5

def _get_fake_prob(model, logits_dict, key="logits"):
    """Extract P(FAKE) using the model's actual class_to_idx mapping."""
    logits = logits_dict[key] / TEMPERATURE
    probs = F.softmax(logits, dim=1)
    c2i = getattr(model, "class_to_idx", {"FAKE": 0, "REAL": 1})
    fake_idx = 0  # default
    for k, v in c2i.items():
        if "fake" in k.lower() or "ai" in k.lower() or "synth" in k.lower():
            fake_idx = v
            break
    return probs[0, fake_idx].item()

def predict_multi_factor(image: Image.Image) -> dict:
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_w, img_h = image.size
    has_face, primary_bbox = detect_faces_and_bboxes(image)
    
    t_full = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    
    # If a valid face is detected, crop it with a 25% margin for the facial forensics stream
    if has_face and primary_bbox is not None:
        fx, fy, fw, fh = primary_bbox
        pad_x = int(fw * 0.25)
        pad_y = int(fh * 0.25)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(img_w, fx + fw + pad_x)
        y2 = min(img_h, fy + fh + pad_y)
        face_crop = image.crop((x1, y1, x2, y2))
        t_face = PREPROCESS(face_crop).unsqueeze(0).to(DEVICE)
    else:
        t_face = t_full
    
    with torch.no_grad():
        out_face_full = model_face(t_full, return_breakdown=True)
        prob_face_full = _get_fake_prob(model_face, out_face_full, "logits")
        spat_face_full = _get_fake_prob(model_face, out_face_full, "spatial_logits")
        spec_face_full = _get_fake_prob(model_face, out_face_full, "spectral_logits")

        if has_face and primary_bbox is not None:
            out_face_crop = model_face(t_face, return_breakdown=True)
            prob_face_crop = _get_fake_prob(model_face, out_face_crop, "logits")
            spat_face_crop = _get_fake_prob(model_face, out_face_crop, "spatial_logits")
            spec_face_crop = _get_fake_prob(model_face, out_face_crop, "spectral_logits")
            prob_face = max(prob_face_crop, prob_face_full)
            spat_face = max(spat_face_crop, spat_face_full)
            spec_face = max(spec_face_crop, spec_face_full)
        else:
            prob_face = prob_face_full
            spat_face = spat_face_full
            spec_face = spec_face_full
            
        # Corroborate with facial tampering specialist if active
        if has_face and model_face_tamper is not None:
            out_tamper = model_face_tamper(t_full, return_breakdown=True)
            prob_tamper = _get_fake_prob(model_face_tamper, out_tamper, "logits")
            prob_face = max(prob_face, prob_tamper)
        
        out_ai = model_ai(t_full, return_breakdown=True)
        prob_ai = _get_fake_prob(model_ai, out_ai, "logits")
        spat_ai = _get_fake_prob(model_ai, out_ai, "spatial_logits")
        spec_ai = _get_fake_prob(model_ai, out_ai, "spectral_logits")

    if USE_PURE_CNN or clip_model is None:
        clip_ai_prob = 0.50
    else:
        clip_ai_prob = clip_ai_probability(image)

    # ── Multi-Factor Decision Fusion ──────────────────────────────────────
    max_spec = max(spec_face, spec_ai)
    best_ai_cnn = max(prob_ai, max_spec)
    
    # Blended generative AI score incorporating Spatial CNN, Spectral 2D FFT, and CLIP
    gen_score = best_ai_cnn * 0.40 + clip_ai_prob * 0.60

    # TIER 1: Clear Deepfake Face Manipulation (FaceForensics++ signature)
    # Strictly requires an actual detected face in the image!
    if has_face and prob_face >= 0.60:
        verdict = "Deepfake"
        label = "Deepfake (Manipulated Face)"
        is_ai = True
        confidence = prob_face
        expert = "Face Forensics Dual-Stream"
        blended_ai = prob_face

    # Decisive Real Human Face Confirmation:
    elif has_face and prob_face <= 0.30 and clip_ai_prob <= 0.55:
        verdict = "Real"
        label = "Real Camera Photo"
        is_ai = False
        confidence = 1.0 - prob_face
        confidence = min(max(confidence, 0.60), 0.98)
        blended_ai = prob_face
        expert = "VeriSight Dual-Stream Face Forensics"

    # TIER 2: AI-Generated Synthetic Content
    # Handles synthetic landscapes, objects, abstract art, or full-frame AI generated faces.
    # Calibrated decision rules:
    # 1. CLIP semantic verifier detects AI synthesis patterns (clip >= 0.54)
    # 2. Cross-domain agreement between CLIP and 2D FFT (clip >= 0.50 and max_spec >= 0.50)
    # 3. For images with NO face: strong 2D FFT periodic upsampling grid harmonics (max_spec >= 0.55)
    # 4. Multi-factor ensemble blended score indicates AI (gen_score >= 0.52)
    # 5. Dual-Stream face detector captures subtle facial synthesis (has_face and prob_face >= 0.55)
    elif (clip_ai_prob >= 0.54) or \
         (clip_ai_prob >= 0.50 and max_spec >= 0.50) or \
         (not has_face and max_spec >= 0.55) or \
         (gen_score >= 0.52) or \
         (has_face and prob_face >= 0.55):
        verdict = "AI-Generated"
        label = "AI-Generated Image"
        is_ai = True
        confidence = max(gen_score, clip_ai_prob)
        blended_ai = confidence
        if clip_ai_prob >= 0.50 and max_spec >= 0.50:
            expert = "2D FFT Spectral + CLIP Agreement"
        elif clip_ai_prob >= 0.54:
            expert = "CLIP Zero-Shot Semantic Verifier"
        elif not has_face and max_spec >= 0.55:
            expert = "2D FFT Spectral Frequency Stream"
        elif has_face and prob_face >= 0.55:
            expert = "Dual-Stream Face Forensics"
        else:
            expert = "Multi-Factor Synthesis Ensemble"

    # TIER 3: Authentic Real Content Fallback
    else:
        verdict = "Real"
        label = "Real Camera Photo"
        is_ai = False
        confidence = 1.0 - gen_score
        confidence = min(max(confidence, 0.55), 0.96)
        blended_ai = gen_score
        expert = "Multi-Factor Real Verification"

    # Compute final confidence percentage
    conf_pct = round(min(max(confidence, 0.0), 1.0) * 100, 1)
    ai_probability = round(min(max(blended_ai, 0.0), 1.0), 4)
    
    return {
        "verdict": verdict,
        "label": label,
        "is_ai": is_ai,
        "confidence_pct": conf_pct,
        "ai_probability": ai_probability,
        # These metrics describe the generative AI expert consistently;
        # face-forensics probabilities are exposed separately in breakdown.
        "spatial_ai_probability": round(spat_ai, 4),
        "spectral_ai_probability": round(spec_ai, 4),
        "has_face": has_face,
        "expert": expert,
        "breakdown": {
            "face_forensic_prob": round(prob_face * 100, 1),
            "face_spatial_fake_prob": round(spat_face * 100, 1),
            "face_spectral_fake_prob": round(spec_face * 100, 1),
            "generative_ai_prob": round(prob_ai * 100, 1),
            "clip_ai_prob": round(clip_ai_prob * 100, 1),
            "spectral_ai_prob": round(spec_ai * 100, 1),
        }
    }


def load_image_from_url(url: str, timeout: float = 5.0) -> Image.Image:
    if url.startswith("data:"):
        return load_image_from_base64(url)
    if url.startswith("file:"):
        from urllib.parse import unquote, urlparse
        parsed = urlparse(url)
        filepath = unquote(parsed.path)
        if filepath.startswith("/") and len(filepath) > 2 and filepath[2] == ":":
            filepath = filepath[1:]
        return Image.open(filepath).convert("RGB")
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        img_bytes = response.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def load_image_from_base64(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "device": str(DEVICE),
        "trained_weights_loaded": is_trained_weights_loaded,
        "mode": "multi_factor_ensemble_v2",
        "face_model_active": model_face is not None,
        "genai_model_active": model_ai is not None,
        "clip_active": clip_model is not None,
        "face_detector": "mediapipe" if mp_face_detector is not None else "haar_cascade",
        "cache_size": len(URL_CACHE.cache)
    })


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    
    image = None
    cache_key = None
    is_dynamic_endpoint = False
    
    if "image_url" in data and data["image_url"]:
        raw_url = str(data["image_url"]).strip()
        url_lower = raw_url.lower()
        if "thispersondoesnotexist" in url_lower or "random" in url_lower or "generate" in url_lower:
            is_dynamic_endpoint = True

        if not is_dynamic_endpoint:
            cache_key = raw_url
            cached = URL_CACHE.get(cache_key)
            if cached:
                cached_copy = dict(cached)
                cached_copy["cached"] = True
                cached_copy["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
                return jsonify(cached_copy)
        try:
            image = load_image_from_url(raw_url)
        except Exception as e:
            return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
            
    elif "image_base64" in data and data["image_base64"]:
        try:
            image = load_image_from_base64(data["image_base64"])
        except Exception as e:
            return jsonify({"error": f"Failed to parse base64 image: {str(e)}"}), 400
            
    elif "file" in request.files:
        try:
            image = Image.open(request.files["file"].stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Failed to read uploaded file: {str(e)}"}), 400
    else:
        return jsonify({"error": "Missing image_url, image_base64, or file"}), 400
        
    result = predict_multi_factor(image)
    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    result["processing_time_ms"] = elapsed_ms
    result["cached"] = False
    
    # Stealth Sightengine auxiliary cross-validation (Strictly local, enabled only via .env)
    if is_sightengine_available():
        try:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            se_res = check_image_sightengine(buf.getvalue())
            if se_res:
                result["sightengine_aux"] = se_res
        except Exception:
            pass

    if cache_key and not is_dynamic_endpoint:
        URL_CACHE.put(cache_key, result)
        
    return jsonify(result)




@app.route("/detect/image", methods=["POST"])
def detect_image_compat():
    # Direct alias for frontend compatibility with reference project
    return predict()


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    urls = data.get("urls", [])
    
    if not isinstance(urls, list):
        return jsonify({"error": "'urls' must be a list"}), 400
        
    if len(urls) == 0:
        return jsonify({"results": {}, "total_processed": 0, "total_elapsed_ms": 0.0})
        
    results = {}
    to_fetch_urls = []
    
    for url in urls[:12]:
        cached = URL_CACHE.get(url)
        if cached:
            res = dict(cached)
            res["cached"] = True
            results[url] = res
        else:
            to_fetch_urls.append(url)
            
    for url in to_fetch_urls:
        try:
            img = load_image_from_url(url, timeout=2.5)
            pred = predict_multi_factor(img)
            pred["cached"] = False
            URL_CACHE.put(url, pred)
            results[url] = pred
        except Exception as e:
            results[url] = {"error": f"Failed to load: {str(e)}", "is_ai": False, "label": "Error"}
            
    total_elapsed_ms = round((time.time() - start_time) * 1000, 1)
    
    return jsonify({
        "results": results,
        "total_processed": len(urls),
        "total_elapsed_ms": total_elapsed_ms
    })


@app.route("/", methods=["GET"])
@app.route("/demo", methods=["GET"])
def demo_page():
    demo_path = PROJECT_ROOT / "demo.html"
    if not demo_path.exists():
        demo_path = PROJECT_ROOT / "extension" / "demo.html"
    if not demo_path.exists():
        demo_path = PROJECT_ROOT / "extension" / "test_page.html"
    return send_file(demo_path)


@app.route("/test", methods=["GET"])
def test_bench_page():
    return send_file(PROJECT_ROOT / "extension" / "test_page.html")


@app.route("/test_images/<path:filename>", methods=["GET"])
def serve_test_image(filename):
    return send_from_directory(PROJECT_ROOT / "extension" / "test_images", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting VeriSight Multi-Factor Forensic API v2.0 on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, load_dotenv=False)
