import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, target_layer, input_tensor, target_class=None):
    """
    Generates a Grad-CAM visualization for a given PyTorch model.
    Used for explainability in Cow Identification and Behavior Analysis.
    
    Args:
        model: The PyTorch model.
        target_layer: The layer to extract gradients from (e.g., model.backbone.blocks[-1].norm1 for ViT).
        input_tensor: Tensor of shape (1, C, H, W).
        target_class: Int, target class index. If None, uses the highest scoring class.
    
    Returns:
        cam_image: A numpy array representing the heatmap overlaid on the image.
    """
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer]) # use_cuda automatically handled if device is GPU
    
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
        
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Convert input tensor to valid image for overlay
    # Reverse ImageNet normalization
    img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Overlay heatmap
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return visualization

def extract_attention_maps(model, input_tensor):
    """
    Extracts attention weights from Transformer layers.
    Specifically useful for Multi-Modal Cross-Attention.
    """
    model.eval()
    with torch.no_grad():
        # Assuming the model returns attention weights as part of its forward pass
        # e.g., output, fused_rep, attn_weights = model(visual, sensor)
        pass # Implementation depends on exactly how model returns them.
