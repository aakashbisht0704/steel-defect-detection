# SteelSight Architecture & Deep Learning Pipeline

## Overview

**SteelSight** is an industrial-grade end-to-end computer vision platform designed for real-time defect inspection and quality assurance on manufactured steel surfaces. The system integrates deep convolutional architectures, explainable AI (Grad-CAM), and a modern dual-interface web stack (FastAPI + React 19 / Streamlit).

---

## 🏗️ High-Level System Architecture

```
                  +----------------------------------+
                  |  Industrial Line / Camera Stream  |
                  | (High-speed Line Scan / GigE UI) |
                  +-----------------+----------------+
                                    |
                                    v
                     +-------------------------------+
                     |   Preprocessing & Enhancement  |
                     |   - CLAHE Contrast Equalization|
                     |   - Bilateral Filtering / Denoise |
                     |   - Resize (224x224) & Normalization |
                     +--------------+----------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
    +------------------------------+  +-------------------------------+
    |  Supervised Classification   |  |   Unsupervised Autoencoder    |
    |  - EfficientNet-B0 (Edge)    |  |   - ConvAutoencoder           |
    |  - ResNet-50 (Transfer Learn)|  |   - Reconstruction Error Map  |
    |  - Baseline CNN (Benchmarking)  - Anomaly Scoring             |
    +---------------+--------------+  +---------------+---------------+
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
                     +-------------------------------+
                     | Explainability Engine (XAI)   |
                     | - Grad-CAM Saliency Maps      |
                     | - Spatial Defect Localization |
                     | - Attention Coverage Ratio    |
                     +--------------+----------------+
                                    |
                                    v
                     +-------------------------------+
                     |  Industrial Severity Engine   |
                     |  - Metallurgical Risk Matrix  |
                     |  - Confidence & Coverage Scale|
                     |  - Grade: Low / Med / Critical|
                     +--------------+----------------+
                                    |
                                    v
                     +-------------------------------+
                     |      FastAPI Engine Layer     |
                     | - High-throughput REST API    |
                     | - Async Image Parsing         |
                     | - Base64 Heatmap Serialization|
                     +-------+---------------+-------+
                             |               |
              +--------------+               +--------------+
              v                                             v
+-------------------------------+             +-------------------------------+
| SteelSight Frontend (React 19)|             |   Streamlit Inspection Lab    |
| - Vite + Tailwind CSS 4       |             | - Fast Python Prototyping     |
| - TanStack Router & Lucide    |             | - Checkpoint Switcher         |
| - Live Inspection & Auditing  |             | - Grad-CAM Visualizer         |
+-------------------------------+             +-------------------------------+
```

---

## 🧠 Model Architectures & Transfer Learning

SteelSight implements and benchmarks four complementary architectures:

### 1. EfficientNet-B0 (Production Deployed)
- **Key Characteristic**: Compound scaling balancing network depth, width, and resolution.
- **Parameters**: `~4.02M` (6x fewer than ResNet-50).
- **Latency**: `~0.55 ms/image` (2.2x faster than ResNet-50).
- **Accuracy**: `99.63%` on holdout test set (`Macro F1: 99.63%`).
- **Classification Head**:
  ```python
  nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(in_features=1280, out_features=6)
  )
  ```
- **Target CAM Layer**: `model.features[-1]` (top convolutional activation map).

### 2. ResNet-50 (Deep Residual Benchmark)
- **Key Characteristic**: 50-layer deep residual network with skip connections to mitigate vanishing gradients.
- **Parameters**: `~24.03M`.
- **Latency**: `~1.23 ms/image`.
- **Accuracy**: `99.63%` (`Macro F1: 99.63%`).
- **Fine-Tuning Strategy**: 
  - *Stage 1*: Backbone frozen, train custom head (Dropout 0.3 -> Linear 256 -> BatchNorm -> ReLU -> Dropout 0.2 -> Linear 6).
  - *Stage 2*: Unfreeze `layer3` and `layer4`, fine-tune end-to-end with low learning rate (`1e-4` with cosine annealing).
- **Target CAM Layer**: `model.layer4[-1]`.

### 3. Baseline CNN (Scratch Control)
- **Key Characteristic**: 3 convolutional blocks with Batch Normalization, Max Pooling, and Adaptive Average Pooling.
- **Parameters**: `~0.36M`.
- **Latency**: `~0.14 ms/image`.
- **Accuracy**: `68.15%` (`Macro F1: 68.53%`).
- Demonstrates the critical importance of pretrained feature representations on texture and edge discrimination.

### 4. Convolutional Autoencoder (Anomaly Detection)
- **Architecture**: 3-stage symmetric encoder-decoder.
- **Objective**: Reconstruction of defect-free steel surfaces.
- **Inference**: High pixel-wise Mean Squared Error (MSE) identifies novel, out-of-distribution defects without requiring explicit class labels.

---

## 🔍 Explainability Engine: Grad-CAM

In manufacturing environments, "black-box" predictions are unacceptable for safety audits. SteelSight integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

1. **Feature Map Extraction**: Computes forward pass activations $A^k$ at the final convolutional feature layer.
2. **Gradient Computation**: Computes the gradient of the predicted class score $y^c$ with respect to feature activation maps:
   $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Heatmap Generation**: Weights and sums feature maps, passing through ReLU to isolate positively correlated defect cues:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
4. **Attention Coverage Ratio**: Calculates the percentage of surface area exhibiting high model attention:
   $$\text{Coverage} = \frac{\sum (L_{\text{norm}} > 0.5)}{H \times W}$$

---

## ⚙️ Industrial Severity Rule Engine

SteelSight couples model predictions with domain-specific metallurgical risk criteria:

| Defect Class | Metallurgical Hazard | Structural Risk | Base Severity |
| :--- | :--- | :--- | :--- |
| **Crazing (Cr)** | High tensile fatigue risk | Propagates into structural fractures | **CRITICAL** |
| **Pitted Surface (PS)**| Localized wall thinning | Accelerates stress corrosion | **CRITICAL** |
| **Inclusion (In)** | Internal non-metallic inclusions | Delamination during deep drawing | **MEDIUM** |
| **Rolled-in Scale (RS)**| Embedded iron oxide mill scale | Surface adhesion & coating failure | **MEDIUM** |
| **Scratches (Sc)** | Surface abrasion gouges | Minor cosmetic / fatigue notch | **LOW / MEDIUM** |
| **Patches (Pa)** | Localized oxidation plates | Aesthetic defect | **LOW** |

A dynamic formula adjusts severity based on model confidence and spatial attention coverage:
$$\text{Risk Score} = \text{Base Risk} \times \text{Confidence} \times (1 + \text{Coverage})$$
- `Score >= 0.70` $\rightarrow$ **CRITICAL** (Automated rejection / line stoppage alert)
- `0.40 <= Score < 0.70` $\rightarrow$ **MEDIUM** (Secondary manual inspection required)
- `Score < 0.40` $\rightarrow$ **LOW** (Acceptable within standard tolerance)
