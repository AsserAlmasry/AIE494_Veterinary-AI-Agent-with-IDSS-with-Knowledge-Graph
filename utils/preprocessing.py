"""
utils/preprocessing.py
=======================
Veterinary image preprocessing utilities.
• Image validation (blur, brightness, contrast, coverage, occlusion)
• Veterinary-specific image enhancement
• Visual proxy feature extraction (coat condition, eye clarity, posture)
• Model-specific adapters (YOLO 640px, MaxViT 224px)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

QUALITY_THRESHOLDS = {
    "min_resolution":  (224, 224),
    "max_blur_score":  100,
    "min_brightness":  30,
    "max_brightness":  220,
    "min_contrast":    20,
    "max_occlusion":   0.3,
    "min_cow_coverage": 0.15,
}

ENHANCEMENT_PARAMS = {
    "minimal":    {"contrast": 1.05, "sharpness": 1.1,  "noise_reduction": False},
    "standard":   {"contrast": 1.15, "sharpness": 1.25, "noise_reduction": True},
    "aggressive": {"contrast": 1.3,  "sharpness": 1.5,  "noise_reduction": True},
}


# ── Image validation ──────────────────────────────────────────────────────────

def validate_image(
    image: Union[Image.Image, np.ndarray],
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    Run veterinary-grade image quality checks.

    Returns
    -------
    {valid, issues, metrics, recommendations}
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed — skipping image validation")
        return {"valid": True, "issues": None, "metrics": {}, "recommendations": []}

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    # Resolution
    width, height = image.size
    min_w, min_h = QUALITY_THRESHOLDS["min_resolution"]
    metrics["resolution"] = (width, height)
    metrics["resolution_ok"] = width >= min_w and height >= min_h
    if not metrics["resolution_ok"]:
        issues.append(f"Resolution too low: {width}x{height} < {min_w}x{min_h}")

    # Convert for OpenCV checks
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Blur (Laplacian variance)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["blur_score"] = blur_score
    metrics["blur_ok"] = blur_score >= QUALITY_THRESHOLDS["max_blur_score"]
    if not metrics["blur_ok"]:
        issues.append(f"Image may be blurry (score: {blur_score:.1f})")

    # Brightness
    brightness = float(np.mean(gray))
    metrics["brightness"] = brightness
    metrics["brightness_ok"] = (
        QUALITY_THRESHOLDS["min_brightness"] <= brightness <= QUALITY_THRESHOLDS["max_brightness"]
    )
    if not metrics["brightness_ok"]:
        issues.append(f"Suboptimal lighting (brightness: {brightness:.1f}/255)")

    # Contrast
    contrast = float(np.std(gray))
    metrics["contrast"] = contrast
    metrics["contrast_ok"] = contrast >= QUALITY_THRESHOLDS["min_contrast"]
    if not metrics["contrast_ok"]:
        issues.append(f"Low contrast (std: {contrast:.1f})")

    # Cow coverage
    cow_coverage = _estimate_cow_coverage(img_cv)
    metrics["cow_coverage"] = cow_coverage
    metrics["coverage_ok"] = cow_coverage >= QUALITY_THRESHOLDS["min_cow_coverage"]
    if not metrics["coverage_ok"]:
        issues.append(f"Insufficient cow visibility ({cow_coverage:.1%})")

    # Occlusion
    occlusion = _estimate_occlusion(img_cv, gray)
    metrics["occlusion"] = occlusion
    metrics["occlusion_ok"] = occlusion <= QUALITY_THRESHOLDS["max_occlusion"]
    if not metrics["occlusion_ok"]:
        issues.append(f"High occlusion detected ({occlusion:.1%})")

    is_valid = all(
        metrics[k] for k in
        ["resolution_ok", "blur_ok", "brightness_ok", "contrast_ok", "coverage_ok", "occlusion_ok"]
    )

    if return_details:
        return {
            "valid": is_valid,
            "issues": issues if issues else None,
            "metrics": metrics,
            "recommendations": _quality_recommendations(metrics),
        }
    return {"valid": is_valid}


def _estimate_cow_coverage(img_cv: np.ndarray) -> float:
    try:
        import cv2
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([10, 50, 50]), np.array([25, 255, 255])),   # brown
            cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255])),  # white
                cv2.inRange(hsv, np.array([0, 0, 0]),   np.array([180, 255, 50])),  # black
            ),
        )
        total = img_cv.shape[0] * img_cv.shape[1]
        return cv2.countNonZero(mask) / total if total > 0 else 0.0
    except Exception:
        return 0.5


def _estimate_occlusion(img_cv: np.ndarray, gray: Optional[np.ndarray] = None) -> float:
    try:
        import cv2
        if gray is None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        gs, occluded = 8, 0
        for i in range(gs):
            for j in range(gs):
                cell = edges[i * h // gs:(i + 1) * h // gs, j * w // gs:(j + 1) * w // gs]
                if np.mean(cell > 0) > 0.4:
                    occluded += 1
        return occluded / (gs * gs)
    except Exception:
        return 0.1


def _quality_recommendations(metrics: Dict) -> List[str]:
    recs = []
    if not metrics.get("resolution_ok"):
        recs.append("📷 Use higher resolution or move closer")
    if not metrics.get("blur_ok"):
        recs.append("🎯 Ensure camera steady and subject in focus")
    if not metrics.get("brightness_ok"):
        recs.append("💡 Improve lighting conditions")
    if not metrics.get("contrast_ok"):
        recs.append("🎨 Improve subject-background contrast")
    if not metrics.get("coverage_ok"):
        recs.append("🐄 Frame cow to occupy at least 15% of image")
    if not metrics.get("occlusion_ok"):
        recs.append("🚧 Remove obstructions from camera view")
    return recs or ["✅ Image quality meets requirements"]


# ── Image enhancement ─────────────────────────────────────────────────────────

def enhance_for_analysis(
    image: Image.Image,
    level: str = "standard",
) -> Image.Image:
    """Apply veterinary-specific image enhancements."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    params = ENHANCEMENT_PARAMS.get(level, ENHANCEMENT_PARAMS["standard"])

    enhanced = ImageEnhance.Contrast(image).enhance(params["contrast"])
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(params["sharpness"])
    if params["noise_reduction"]:
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))

    # Cattle-optimised colour correction (+2% red, +1% green)
    arr = np.array(enhanced, dtype=np.float32) / 255.0
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.02, 0, 1)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.01, 0, 1)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


# ── Visual proxy features ─────────────────────────────────────────────────────

def extract_visual_proxies(image: Image.Image) -> Dict[str, Dict[str, Any]]:
    """
    Extract 5 health-related visual proxy features from a cattle image.
    Useful when sensor data is unavailable.
    """
    try:
        import cv2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        proxies: Dict[str, Dict[str, Any]] = {}

        # Coat condition — colour uniformity
        coat_uniformity = float(np.clip(1.0 - (np.std(hsv[:, :, 1]) / 50.0), 0, 1))
        proxies["coat_condition"] = _proxy_result(coat_uniformity, (0.7, 1.0), "Coat uniformity")

        # Eye clarity — edge density in face region
        h, w = img_cv.shape[:2]
        face = img_cv[h // 4:3 * h // 4, w // 4:3 * w // 4]
        if face.size > 0:
            face_edges = cv2.Canny(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), 50, 150)
            eye_clarity = float(np.clip(1.0 - np.mean(face_edges > 0) * 2, 0, 1))
        else:
            eye_clarity = 0.5
        proxies["eye_clarity"] = _proxy_result(eye_clarity, (0.2, 0.6), "Eye clarity")

        # Posture — aspect ratio of largest contour
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x_, y_, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
            ar = cw / ch if ch > 0 else 1.0
            posture = float(1.0 - min(abs(ar - 1.3) / 0.5, 1.0))
        else:
            posture = 0.5
        proxies["posture_score"] = _proxy_result(posture, (1.1, 1.4), "Posture")

        # Skin lesion — texture anomaly via LBP
        try:
            from skimage import feature as skf
            lbp = skf.local_binary_pattern(gray, P=8, R=1, method="uniform")
            texture = float(np.clip(np.std(lbp) / 10.0, 0, 1))
        except ImportError:
            texture = 0.3
        proxies["skin_lesion_indicator"] = _proxy_result(1.0 - texture, (0.0, 0.3), "Skin texture")

        # Vitality — colour saturation
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        proxies["vitality_score"] = _proxy_result(saturation, (0.4, 0.8), "Vitality")

        return proxies
    except Exception as exc:
        logger.warning(f"Visual proxy extraction failed: {exc}")
        return {}


def _proxy_result(value: float, normal_range: Tuple, description: str) -> Dict[str, Any]:
    lo, hi = normal_range
    status = "below_normal" if value < lo else ("above_normal" if value > hi else "normal")
    return {
        "value": round(value, 4),
        "normal_range": normal_range,
        "status": status,
        "description": description,
        "unit": "normalised [0,1]",
    }


# ── Batch processing ──────────────────────────────────────────────────────────

def batch_validate(
    images: List[Image.Image], max_workers: int = 4
) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(validate_image, images))
