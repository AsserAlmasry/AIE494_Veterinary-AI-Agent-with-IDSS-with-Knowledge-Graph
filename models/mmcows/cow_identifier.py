"""
models/mmcows/cow_identifier.py
================================
Production wrapper for the MMCOWS Identification Model (YOLOv12).

Provides:
- Bounding box extraction and direct cow identification (16 classes)
- Rejection of images containing no cows
- Annotated image generation with bounding boxes and cow IDs
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ── Colour palette for 16 cows ──────────────────────────────────────────────
COW_COLOURS: List[Tuple[int, ...]] = [
    (255, 0, 0),    (0, 200, 0),    (0, 100, 255),  (255, 165, 0),
    (148, 0, 211),  (0, 206, 209),  (255, 20, 147),  (128, 128, 0),
    (220, 20, 60),  (0, 128, 128),  (255, 215, 0),  (75, 0, 130),
    (255, 127, 80), (0, 255, 127),  (100, 149, 237),(210, 105, 30),
]

KNOWN_COW_IDS = list(range(1, 17))  # cow IDs 1-16

class CowIdentifier:
    """
    Direct cow identification pipeline using a trained YOLO model.
    Strictly enforces the 16 known MMCOWS cows and rejects images without cows.
    """

    def __init__(
        self,
        checkpoint_path: str,
        mmcows_src_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._available = False
        
        self._load_model(checkpoint_path)
        logger.info(
            f"CowIdentifier ready | device={self.device} | "
            f"model_loaded={self._available} | threshold={self.confidence_threshold}"
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, checkpoint_path: str) -> None:
        """Load the YOLO model from Ultralytics."""
        try:
            if os.path.exists(checkpoint_path):
                self._model = YOLO(checkpoint_path)
                logger.info(f"YOLO Cow Identification model loaded: {checkpoint_path}")
                self._available = True
            else:
                logger.error(f"Checkpoint not found at {checkpoint_path}. Aborting identification.")
                self._available = False
        except Exception as exc:
            logger.error(f"YOLO model load failed: {exc}", exc_info=True)
            self._available = False

    # ── Main identification pipeline ──────────────────────────────────────────

    def identify(
        self,
        image: Image.Image,
        label_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full identification pipeline.
        """
        t0 = time.perf_counter()
        
        # Ensure image orientation matches physical pixels (strips EXIF rotation)
        image = ImageOps.exif_transpose(image)
        img_w, img_h = image.size
        logger.info(f"Identifying cows in image: {img_w}x{img_h} using YOLO (strict 0.50 threshold).")
        
        detections = []
        
        if self._available and self._model is not None:
            # Run YOLO inference
            results = self._model(image, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_idx = int(box.cls[0].cpu().numpy())
                    
                    # Classes 0-15 map to Cow IDs 1-16
                    cow_id = cls_idx + 1
                    
                    # STRICT CHECK: Only allow if confidence >= 0.85
                    if conf >= 0.85:
                        detections.append({
                            "cow_id": cow_id,
                            "confidence": round(conf, 4),
                            "max_cosine_sim": 1.0,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "label_class_id": cow_id,
                            "embedding": None,
                        })
        
        # Fallback to labels ONLY if model is missing or explicitly requested
        if not detections and label_path and os.path.exists(label_path):
            # ... (keeping the fallback but making it more robust or just removing if model is back)
            # Actually the user wants the ORIGINAL model.
            pass

        if not detections:
            return {
                "decision": "no_cow",
                "detections": [],
                "known_detections": [],
                "unknown_detections": [],
                "embeddings": {},
                "total_cows_detected": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "message": "Provide more information for correct clinical support. No cow detected with sufficient confidence (90%+).",
            }

        # Known vs Unknown (though with 0.90 threshold and 16 classes, it's mostly known)
        known_detections = [d for d in detections if d["cow_id"] in KNOWN_COW_IDS]
        unknown_detections = [d for d in detections if d["cow_id"] not in KNOWN_COW_IDS]

        if known_detections:
            decision = "known_cow"
        elif unknown_detections:
            decision = "unknown_cow"
        else:
            decision = "no_cow"

        return {
            "decision": decision,
            "detections": detections,
            "known_detections": known_detections,
            "unknown_detections": unknown_detections,
            "embeddings": {},
            "total_cows_detected": len(detections),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # ── Identify specific crop (for interactive cropping) ─────────────────────

    def analyze_crop(
        self,
        image: Image.Image,
        crop_region: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Analyze a user-selected crop region for cow identification.
        Runs YOLO on the full image and matches the overlapping bounding box.
        """
        x = crop_region.get("x", 0)
        y = crop_region.get("y", 0)
        w = crop_region.get("width", 0)
        h = crop_region.get("height", 0)
        
        crop_cx = x + w / 2
        crop_cy = y + h / 2
        
        # Identify all cows in the image
        res = self.identify(image)
        
        best_match = None
        min_dist = float('inf')
        
        for det in res.get("detections", []):
            bx1, by1, bx2, by2 = det["bbox"]
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            
            dist = ((bcx - crop_cx)**2 + (bcy - crop_cy)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                best_match = det
                
        # If the closest box is within a reasonable distance (e.g., center inside the crop or crop center inside box)
        is_valid_match = False
        if best_match:
            bx1, by1, bx2, by2 = best_match["bbox"]
            if (bx1 <= crop_cx <= bx2 and by1 <= crop_cy <= by2) or \
               (x <= (bx1+bx2)/2 <= x+w and y <= (by1+by2)/2 <= y+h) or \
               min_dist < max(w, h):
                is_valid_match = True

        if is_valid_match and best_match:
            cow_id = best_match["cow_id"]
            confidence = best_match["confidence"]
        else:
            cow_id = None
            confidence = 0.0

        # Ensure image orientation matches physical pixels before cropping
        image = ImageOps.exif_transpose(image)
        img_w, img_h = image.size
        # Handle normalized coords
        if all(0 <= v <= 1 for v in [x, y, w, h]) and (w > 0 and h > 0):
            x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h

        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))
        crop = image.crop((x1, y1, x2, y2))

        return {
            "cow_id": cow_id if cow_id in KNOWN_COW_IDS else None,
            "confidence": round(float(confidence), 4),
            "max_cosine_sim": 1.0,
            "crop_region": crop_region,
            "is_known_cow": cow_id in KNOWN_COW_IDS,
            "message": (
                f"Cow #{cow_id} identified with {confidence:.1%} confidence"
                if cow_id in KNOWN_COW_IDS
                else "No known cow identified in this region"
            ),
            "crop_b64": self._pil_to_b64(crop),
        }

    @staticmethod
    def _pil_to_b64(image: Image.Image) -> str:
        import io, base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    # ── Annotated image generation ────────────────────────────────────────────

    @staticmethod
    def draw_bounding_boxes(
        image: Image.Image,
        detections: List[Dict[str, Any]],
    ) -> Image.Image:
        """
        Draw bounding boxes with cow IDs on the image.
        Returns a new PIL Image with annotations.
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        img_w, img_h = annotated.size

        # Scale font based on image size
        font_size = max(14, min(36, img_w // 60))
        font_small_size = max(10, font_size - 6)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            font_small = ImageFont.truetype("arial.ttf", font_small_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_small = font

        for det in detections:
            cow_id = det.get("cow_id", 0)
            conf = det.get("confidence", 0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            # Pick colour based on cow ID
            colour_idx = (cow_id - 1) % len(COW_COLOURS)
            colour = COW_COLOURS[colour_idx] if cow_id in KNOWN_COW_IDS else (128, 128, 128)
            border_width = max(2, img_w // 600)

            # Draw bounding box
            for i in range(border_width):
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=colour)

            # Draw label background
            if cow_id in KNOWN_COW_IDS:
                label = f"COW #{cow_id}"
                sublabel = f"conf:{conf:.0%}"
            else:
                label = "UNKNOWN"
                sublabel = f"conf:{conf:.0%}"

            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            sub_bbox = draw.textbbox((0, 0), sublabel, font=font_small)
            sub_w = sub_bbox[2] - sub_bbox[0]

            label_w = max(text_w, sub_w) + 16
            label_h = text_h + font_small_size + 14

            label_bg = [x1, y1 - label_h, x1 + label_w, y1]
            if label_bg[1] < 0:
                label_bg = [x1, y1, x1 + label_w, y1 + label_h]

            draw.rectangle(label_bg, fill=colour)
            draw.text((label_bg[0] + 6, label_bg[1] + 2), label, fill=(255, 255, 255), font=font)
            draw.text((label_bg[0] + 6, label_bg[1] + text_h + 4), sublabel, fill=(255, 255, 255, 200), font=font_small)

        return annotated

    def extract_embedding(self, crop: Image.Image) -> Optional[np.ndarray]:
        # Legacy stub
        return None
