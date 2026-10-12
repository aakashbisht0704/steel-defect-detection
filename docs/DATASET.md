# NEU Surface Defect Database Guide

The **NEU Surface Defect Database** (developed by Northeastern University, China) is a premier benchmark dataset for evaluating automated visual inspection systems on hot-rolled steel strips.

---

## 📊 Dataset Overview

- **Total Samples**: 1,800 images
- **Number of Classes**: 6 balanced categories (300 images per category)
- **Native Resolution**: $200 \times 200$ pixels
- **Color Format**: Grayscale (8-bit depth)
- **Standard Split**: 70% Train (1,260 images) / 15% Validation (270 images) / 15% Test (270 images), stratified by class.

---

## 🔬 Defect Categories & Metallurgical Background

| Code | Defect Class | Visual Morphology | Metallurgical Cause | Structural Severity |
| :--- | :--- | :--- | :--- | :--- |
| **Cr** | **Crazing** | Complex, multi-directional network of fine micro-cracks resembling a web. | Excessive surface cooling rate, thermal stress during rolling, or excessive copper content. | **CRITICAL**: Can propagate under cyclic mechanical stress, leading to structural fatigue failure. |
| **In** | **Inclusion** | Non-metallic spots, dots, or tail-like dark embedded particles. | Entrapped slag, deoxidation products ($Al_2O_3$, $SiO_2$), or refractory lining erosion in the mold. | **MEDIUM**: Initiates subsurface voids during deep drawing and cold stamping operations. |
| **Pa** | **Patches** | Irregular, plate-like darker or lighter surface patches with uneven borders. | Uneven scale formation, localized chemical segregation, or slab reheating non-uniformity. | **LOW**: Primarily visual/aesthetic; slight impact on uniform paint and zinc coating adhesion. |
| **PS** | **Pitted Surface** | Dense cluster of small craters, holes, and localized indentations. | Mechanical dislodgement of oxidized scale or acid pickling over-etching. | **CRITICAL**: Local wall thickness reduction creates stress concentration points and accelerates corrosion. |
| **RS** | **Rolled-in Scale** | Dark streaks, fish-scale bands, or pressed oxide layers embedded in steel. | High-pressure descaling header failure; furnace scale forced into metal by rolling roll pressure. | **MEDIUM / HIGH**: Compromises surface passivation, galvanization, and anti-corrosive coating integrity. |
| **Sc** | **Scratches** | Long, directional mechanical gouges, troughs, or linear abrasions. | Sharp guide rails, misaligned roller tables, or foreign debris contact during transport. | **LOW / MEDIUM**: Cosmetic; severe deep gouges act as mechanical notch stress risers. |

---

## 📥 Automated Download & Setup

SteelSight includes an automated downloader script that fetches and unpacks the verified dataset archive:

```bash
python download_data.py
```

### Manual Setup
If automated download is interrupted, download the dataset zip directly from:
- [Google Drive Mirror](https://drive.google.com/file/d/1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k/view)
- Or [Kaggle NEU Surface Defect Database](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

Place the extracted folders in `data/NEU-CLS/` such that the directory structure matches:

```
data/
└── NEU-CLS/
    ├── crazing/          # 300 images (*.bmp / *.jpg)
    ├── inclusion/        # 300 images
    ├── patches/          # 300 images
    ├── pitted_surface/   # 300 images
    ├── rolled_in_scale/  # 300 images
    └── scratches/        # 300 images
```

---

## 🧪 Preprocessing & Augmentation Pipeline

Industrial inspection settings suffer from lighting variations, camera sensor noise, and roll oil residue. SteelSight applies a robust Albumentations pipeline defined in `src/dataset.py`:

```python
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Amplifies subtle micro-crack boundaries without over-amplifying background noise.
- **RandomRotate90 & Flips**: Enforces rotational invariance since surface defects appear at arbitrary angles relative to rolling direction.
- **Gaussian Noise**: Simulates CMOS/CCD sensor noise under low factory illumination.
