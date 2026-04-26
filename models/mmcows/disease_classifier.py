import os
import io
import time
import logging
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Verify timm is installed
try:
    import timm
except ImportError:
    logger.error("timm is required for the DiseaseClassifier. Please install it.")

class MaxVitDiseaseClassifier(nn.Module):
    """
    Custom architecture reverse-engineered from the user's best_model.pth.
    Uses a maxvit_base_tf_224 backbone with a custom MLP head for 5-class prediction.
    """
    def __init__(self, model_name='maxvit_base_tf_224', num_classes=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.backbone.num_features),
            nn.Dropout(0.5),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class MaxVitDiseaseService:
    """
    Service wrapper for the PyTorch DiseaseClassifier.
    Replaces the dummy Groq Vision service with actual model inference.
    """
    CLASS_NAMES = [
        'Lumpy Skin Disease', 
        'Lameness Disease', 
        'Foot and Mouth Disease', 
        'Mastitis', 
        'Healthy'
    ]

    def __init__(self, model_path: str = r"C:\Users\Dell\Downloads\best_model.pth", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._available = False
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Disease model not found at {self.model_path}")
                return

            self._model = MaxVitDiseaseClassifier(num_classes=len(self.CLASS_NAMES))
            
            # Load state dict
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Use strict=False because BatchNorm running stats might mismatch slightly
            self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            self._model.to(self.device).eval()
            self._available = True
            logger.info(f"MaxVit Disease Classifier successfully loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Disease Classifier: {e}", exc_info=True)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze an image (or crop) and return the predicted disease and confidence.
        Matches the output structure expected by the pipeline.
        """
        t0 = time.perf_counter()
        
        if not self._available or self._model is None:
            return {"error": "Disease model not loaded.", "disease_candidates": []}

        try:
            image = Image.open(io.BytesIO(image_bytes))
            img_tensor = self._preprocess(image)

            with torch.no_grad():
                logits = self._model(img_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
                
            # Get top predictions
            top_prob, top_idx = torch.max(probs, dim=0)
            predicted_class = self.CLASS_NAMES[top_idx.item()]
            confidence = top_prob.item()
            
            # Create candidates list (sort by probability descending)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            candidates = []
            for p, idx in zip(sorted_probs, sorted_indices):
                candidates.append({
                    "disease": self.CLASS_NAMES[idx.item()],
                    "confidence": p.item()
                })

            return {
                "top_prediction": predicted_class,
                "confidence": confidence,
                "disease_candidates": candidates,
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "is_healthy": predicted_class == "Healthy"
            }
            
        except Exception as e:
            logger.error(f"Disease inference failed: {e}", exc_info=True)
            return {"error": str(e), "disease_candidates": []}
