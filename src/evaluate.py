"""
Model evaluation & error analysis script.
Calculates:
- Per-class Precision, Recall, F1-Score
- Macro F1 & Overall Test Accuracy
- Confusion Matrix visualization
- Worst misclassification report
"""
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

from dataset import create_dataloaders
from models import get_resnet50, get_efficientnet_b0, BaselineCNN

def evaluate_model(checkpoint_path: str, data_dir: str = "data/NEU-CLS", batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Evaluating checkpoint: {checkpoint_path} on {device}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "resnet50")
    idx_to_class = ckpt.get("idx_to_class", {})
    num_classes = len(idx_to_class)
    
    # Dataloader
    _, _, test_loader, discovered_idx = create_dataloaders(data_dir=data_dir, batch_size=batch_size)
    if not idx_to_class:
        idx_to_class = discovered_idx
        num_classes = len(idx_to_class)
        
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Model architecture
    if model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, freeze_backbone=False)
    elif model_name == "efficientnet":
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=False)
    else:
        model = BaselineCNN(num_classes=num_classes)
        
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification Report
    print("\n" + "=" * 65)
    print(f"      COMPREHENSIVE TEST EVALUATION REPORT ({model_name.upper()})")
    print("=" * 65)
    report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name.upper()}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Ground Truth", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_cm_path = f"outputs/confusion_matrix_{model_name}.png"
    plt.savefig(out_cm_path, dpi=200)
    plt.close()
    print(f">> Confusion matrix heatmap saved to {out_cm_path}")

    # Save summary report text
    with open(f"outputs/evaluation_report_{model_name}.txt", "w") as f:
        f.write(report_text)

    return all_preds, all_labels, all_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_resnet50.pth")
    parser.add_argument("--data_dir", type=str, default="data/NEU-CLS")
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.data_dir)
