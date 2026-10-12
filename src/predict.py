"""
Inference script to classify an unseen industrial surface image.
Matches Problem Statement Expected Output:
'Predicted Defect: Scratches; Confidence: 91%'
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

from dataset import get_val_transforms
from models import get_resnet50, get_efficientnet_b0, BaselineCNN

def predict_single_image(image_path: str, checkpoint_path: str = "checkpoints/best_resnet50.pth", save_cam: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "resnet50")
    idx_to_class = ckpt.get("idx_to_class", {})
    num_classes = len(idx_to_class) if idx_to_class else 6

    # Load Model
    if model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, freeze_backbone=False)
    elif model_name == "efficientnet":
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=False)
    else:
        model = BaselineCNN(num_classes=num_classes)
        
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Read & Preprocess Image
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Failed to load image from {image_path}")
        
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    transform = get_val_transforms(img_size=224)
    augmented = transform(image=img_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        conf, pred_idx = probs.max(0)
        
    pred_idx = pred_idx.item()
    confidence_pct = conf.item() * 100.0
    pred_class_name = idx_to_class.get(pred_idx, f"Class_{pred_idx}").capitalize()

    # Exact expected output format:
    print(f"Predicted Defect: {pred_class_name}; Confidence: {confidence_pct:.0f}%")
    
    # Detailed breakdown
    print("\nDefect Probability Distribution:")
    for idx, p in enumerate(probs):
        cls_name = idx_to_class.get(idx, f"Class_{idx}").capitalize()
        bar = "#" * int(p.item() * 30)
        print(f"  {cls_name:<16}: {p.item()*100:5.1f}% | {bar}")

    return pred_class_name, confidence_pct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict defect for an industrial surface image")
    parser.add_argument("--image", type=str, required=True, help="Path to input surface image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_resnet50.pth", help="Model checkpoint")
    args = parser.parse_args()

    predict_single_image(args.image, args.checkpoint)
