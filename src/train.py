"""
Training script for Industrial Defect Classification.
Supports:
- ResNet-50 (with 2-stage transfer learning)
- EfficientNet-B0
- BaselineCNN
"""
import os
import argparse
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from dataset import create_dataloaders
from models import get_resnet50, unfreeze_resnet_top_layers, get_efficientnet_b0, BaselineCNN

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        
    return running_loss / total, 100.0 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
    return running_loss / total, 100.0 * correct / total

def plot_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker='s')
    ax1.set_title("Loss vs Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history["train_acc"], label="Train Acc (%)", marker='o')
    ax2.plot(epochs, history["val_acc"], label="Val Acc (%)", marker='s')
    ax2.set_title("Accuracy vs Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train defect classification model")
    parser.add_argument("--data_dir", type=str, default="data/NEU-CLS", help="Path to defect dataset")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet", "baseline"])
    parser.add_argument("--epochs", type=int, default=12, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    print(f">> Loading data from {args.data_dir}...")
    train_loader, val_loader, test_loader, idx_to_class = create_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )
    num_classes = len(idx_to_class)
    print(f">> Classes: {idx_to_class}")

    # Initialize model
    if args.model == "resnet50":
        print(">> Building ResNet-50 (Transfer Learning)...")
        model = get_resnet50(num_classes=num_classes, freeze_backbone=True).to(device)
        optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.model == "efficientnet":
        print(">> Building EfficientNet-B0 (Transfer Learning)...")
        model = get_efficientnet_b0(num_classes=num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
    else:
        print(">> Building BaselineCNN (Trained from scratch)...")
        model = BaselineCNN(num_classes=num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    checkpoint_path = f"checkpoints/best_{args.model}.pth"

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # 2-stage transfer learning: unfreeze ResNet layers at halfway point
        if args.model == "resnet50" and epoch == (args.epochs // 2) + 1:
            print("\n>> [Stage 2] Unfreezing ResNet layer3 and layer4 for fine-tuning...")
            model = unfreeze_resnet_top_layers(model)
            optimizer = optim.AdamW([
                {'params': model.layer3.parameters(), 'lr': 1e-5},
                {'params': model.layer4.parameters(), 'lr': 1e-5},
                {'params': model.fc.parameters(), 'lr': 2e-4},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - epoch + 1))

        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "best_acc": best_val_acc,
                "idx_to_class": idx_to_class
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path} ({v_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n>> Training completed in {total_time:.1f}s. Peak Validation Accuracy: {best_val_acc:.2f}%")

    plot_curves(history, f"outputs/training_curves_{args.model}.png")
    print(f">> Training curves saved to outputs/training_curves_{args.model}.png")

if __name__ == "__main__":
    main()
