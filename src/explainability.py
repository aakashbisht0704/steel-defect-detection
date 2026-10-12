"""
Explainability module: Visualizing defect localization with Grad-CAM.
Generates heatmaps highlighting the exact surface defect regions responsible for classification.
"""
import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from dataset import get_val_transforms
    from models import get_resnet50, get_efficientnet_b0
except ImportError:
    from src.dataset import get_val_transforms
    from src.models import get_resnet50, get_efficientnet_b0

def generate_gradcam(image_path: str, checkpoint_path: str = "checkpoints/best_resnet50.pth", output_path: str = "outputs/gradcam_result.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "resnet50")
    idx_to_class = ckpt.get("idx_to_class", {})
    num_classes = len(idx_to_class) if idx_to_class else 6

    # Load model
    if model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, freeze_backbone=False)
        target_layers = [model.layer4[-1]]
    else:
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=False)
        target_layers = [model.features[-1]]

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Read and resize
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_float = np.float32(img_resized) / 255.0

    transform = get_val_transforms(img_size=224)
    input_tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)
        conf, pred_idx = probs.max(0)
        
    pred_idx = pred_idx.item()
    pred_label = idx_to_class.get(pred_idx, f"Defect_{pred_idx}").capitalize()
    confidence = conf.item() * 100.0

    # Compute Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Overlay
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # Plot & Save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title(f"Original Surface Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="inferno")
    axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(cam_image)
    axes[2].set_title(f"Overlay: {pred_label} ({confidence:.1f}%)", fontsize=12, fontweight='bold', color='darkred')
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f">> Grad-CAM visualization saved to: {output_path}")
    print(f">> Defect: {pred_label}, Confidence: {confidence:.1f}%")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_resnet50.pth")
    parser.add_argument("--output", type=str, default="outputs/gradcam_analysis.png")
    args = parser.parse_args()

    generate_gradcam(args.image, args.checkpoint, args.output)
