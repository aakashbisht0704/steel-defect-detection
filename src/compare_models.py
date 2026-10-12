"""
Model Comparison Script for Industrial Defect Detection.
Compares:
1. ResNet-50 (Transfer Learning)
2. EfficientNet-B0 (Transfer Learning)
3. Baseline CNN (Trained from Scratch)

Evaluates on identical Test split across:
- Overall Accuracy
- Macro F1-Score
- Precision & Recall
- Inference Latency (ms per image)
- Model Size (Parameters & Checkpoint MB)
"""
import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset import create_dataloaders
from models import get_resnet50, get_efficientnet_b0, BaselineCNN

def benchmark_model(model_name: str, ckpt_path: str, test_loader, device):
    if not os.path.exists(ckpt_path):
        print(f"Skipping {model_name}, checkpoint {ckpt_path} not found.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = len(ckpt.get("idx_to_class", {})) or 6

    if model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, freeze_backbone=False)
    elif model_name == "efficientnet":
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=False)
    else:
        model = BaselineCNN(num_classes=num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Model size
    num_params = sum(p.numel() for p in model.parameters())
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

    # Latency and predictions
    all_preds, all_labels = [], []
    latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            latencies.append((time.perf_counter() - t0) / images.size(0))
            
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro") * 100.0
    precision = precision_score(all_labels, all_preds, average="macro") * 100.0
    recall = recall_score(all_labels, all_preds, average="macro") * 100.0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000.0

    return {
        "Model": model_name.upper(),
        "Accuracy (%)": round(acc, 2),
        "Macro F1 (%)": round(macro_f1, 2),
        "Precision (%)": round(precision, 2),
        "Recall (%)": round(recall, 2),
        "Params (M)": round(num_params / 1e6, 2),
        "Checkpoint (MB)": round(ckpt_size_mb, 2),
        "Latency (ms/img)": round(avg_latency_ms, 2)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Running Model Benchmarks on {device}...")

    _, _, test_loader, _ = create_dataloaders("data/NEU-CLS", batch_size=32)

    models_to_test = [
        ("resnet50", "checkpoints/best_resnet50.pth"),
        ("efficientnet", "checkpoints/best_efficientnet.pth"),
        ("baseline", "checkpoints/best_baseline.pth"),
    ]

    results = []
    for m_name, path in models_to_test:
        res = benchmark_model(m_name, path, test_loader, device)
        if res:
            results.append(res)

    if not results:
        print("No checkpoints found. Please train models first using src/train.py.")
        return

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("                    MODEL COMPARISON BENCHMARK TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Save comparison table
    df.to_csv("outputs/model_comparison_benchmark.csv", index=False)
    print("\n>> Benchmark results saved to outputs/model_comparison_benchmark.csv")

    # Plot comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df.plot(x="Model", y=["Accuracy (%)", "Macro F1 (%)"], kind="bar", ax=ax1, colormap="viridis")
    ax1.set_title("Classification Performance Comparison", fontsize=12, fontweight='bold')
    ax1.set_ylim(70, 105)
    ax1.set_ylabel("Score (%)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    df.plot(x="Model", y=["Latency (ms/img)"], kind="bar", ax=ax2, color="coral")
    ax2.set_title("Inference Latency (Lower is Better)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Milliseconds / Image")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("outputs/model_comparison_chart.png", dpi=200)
    plt.close()
    print(">> Benchmark visualization saved to outputs/model_comparison_chart.png")

if __name__ == "__main__":
    main()
