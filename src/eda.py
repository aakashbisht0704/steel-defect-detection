"""
Exploratory Data Analysis (EDA) for Industrial Defect Dataset.
Generates:
1. Class distribution bar plot
2. Visual sample gallery for each defect type
3. Pixel intensity distribution histograms (highlighting surface texture differences)
4. Channel/Resolution statistics
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

def run_eda(data_dir: str = "data/NEU-CLS", output_dir: str = "outputs"):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    data_path = Path(data_dir)
    
    print(f">> Starting Exploratory Data Analysis on {data_dir}...")
    
    # 1. Discover classes and images
    class_images = defaultdict(list)
    for p in data_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".bmp", ".jpg", ".png", ".jpeg"]:
            cls_name = p.parent.name.lower().replace(" ", "_").replace("-", "_")
            class_images[cls_name].append(p)
            
    if not class_images:
        print(f"!! No images found in {data_dir}. Please run download_data.py first.")
        return

    print(f">> Found {len(class_images)} defect categories:")
    for c, imgs in class_images.items():
        print(f"   - {c}: {len(imgs)} samples")

    # 2. Plot Class Balance Distribution
    plt.figure(figsize=(10, 5))
    names = list(class_images.keys())
    counts = [len(class_images[k]) for k in names]
    colors = sns.color_palette("muted", len(names))
    
    bars = plt.bar(names, counts, color=colors, edgecolor='black', alpha=0.85)
    plt.title("NEU Surface Defect Database - Class Balance Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Defect Type", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center', va='bottom')
    plt.ylim(0, max(counts) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_class_distribution.png", dpi=200)
    plt.close()
    print(f">> Class balance chart saved to {output_dir}/eda_class_distribution.png")

    # 3. Defect Category Sample Grid
    fig, axes = plt.subplots(len(names), 4, figsize=(12, 2.5 * len(names)))
    if len(names) == 1:
        axes = np.expand_dims(axes, 0)
        
    for i, cls in enumerate(names):
        imgs = class_images[cls][:4]
        for j, img_path in enumerate(imgs):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].set_title(f"{cls.capitalize()} #{j+1}" if j == 0 else f"Sample #{j+1}", fontsize=10)
            axes[i, j].axis("off")
            
    plt.suptitle("Characteristic Surface Defect Patterns (NEU Benchmark)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_sample_gallery.png", dpi=200)
    plt.close()
    print(f">> Defect gallery saved to {output_dir}/eda_sample_gallery.png")

    # 4. Pixel Intensity Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, cls in enumerate(names[:6]):
        intensities = []
        for img_path in class_images[cls][:40]: # sample 40 images
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            intensities.append(img.flatten())
        data = np.concatenate(intensities)
        axes[i].hist(data, bins=50, color=colors[i % len(colors)], alpha=0.75, density=True)
        axes[i].set_title(f"{cls.capitalize()} (μ={data.mean():.1f}, σ={data.std():.1f})", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Pixel Value (0-255)")
        axes[i].set_ylabel("Density")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Surface Pixel Intensity Profiles Across Defect Categories", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_intensity_profiles.png", dpi=200)
    plt.close()
    print(f">> Pixel intensity profiles saved to {output_dir}/eda_intensity_profiles.png")
    print(">> EDA completed successfully!")

if __name__ == "__main__":
    run_eda()
