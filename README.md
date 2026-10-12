<div align="center">

# 🏭 Steel Defect Detection

**Real-Time Computer Vision & Explainable AI (Grad-CAM) for Hot-Rolled Steel Quality Assurance**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.3-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.3-38B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.63%25-success.svg?style=flat)](#-model-benchmarks--results)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Architecture](docs/ARCHITECTURE.md) • [REST API](docs/API.md) • [Dataset Guide](docs/DATASET.md) • [Deployment](docs/DEPLOYMENT.md) • [Contributing](CONTRIBUTING.md)

</div>

---

## 📌 Executive Summary

In high-speed metallurgical manufacturing, hot-rolled steel strip defects directly degrade tensile strength, corrosion resistance, and fatigue life. Traditional manual visual inspection is subjective, hazardous, and cannot keep pace with modern rolling mills producing coils at up to $20\text{ m/s}$.

**SteelSight** is an end-to-end, edge-ready automated defect inspection platform trained on the **NEU Surface Defect Database**. It pairs state-of-the-art deep convolutional networks with **Grad-CAM visual explainability** and a metallurgical **severity rule engine**, served through a high-performance **FastAPI** backend and an industrial **React 19 HMI dashboard**.

### 🌟 Key Highlights

- **🎯 State-of-the-Art Accuracy**: **99.63% test accuracy** and **99.63% Macro F1** across all 6 NEU surface defect categories.
- **⚡ Ultra-Low Latency Inference**: **0.55 ms/image** forward pass using compound-scaled **EfficientNet-B0** (4.02M parameters, 6x smaller and 2.2x faster than ResNet-50).
- **🔬 Explainable AI (Grad-CAM)**: Real-time visual saliency maps highlighting exact defect regions and calculating attention surface coverage percentages for transparent quality audits.
- **🛡️ Metallurgical Severity Engine**: Automatic risk categorization (**CRITICAL**, **MEDIUM**, **LOW**) based on structural crack propagation risk, model confidence, and defect surface area.
- **💻 Dual Production Interfaces**:
  - **SteelSight React Dashboard**: Modern industrial HMI (React 19, TypeScript, Tailwind CSS, TanStack Router) featuring live drag-and-drop inspection, model comparison visualizer, and inspection history logs.
  - **Streamlit Inspection Lab**: Rapid Python dashboard for model checkpoint switching and interactive analysis.
- **🚀 Production REST API**: Asynchronous FastAPI service returning predicted defect, confidence distribution, base64 Grad-CAM overlay, and hardware latency.
- **🔍 Anomaly Detection Support**: Includes an unsupervised Convolutional Autoencoder baseline for out-of-distribution surface flaw detection.

---

## 📊 Model Benchmarks & Results

Evaluated on the holdout test set (270 images, balanced across all 6 classes):

| Model | Accuracy (%) | Macro F1 (%) | Precision (%) | Recall (%) | Parameters | Checkpoint | Latency (GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Deployed)** 🚀 | **99.63%** | **99.63%** | **99.64%** | **99.63%** | **4.02M** | **15.6 MB** | **0.55 ms/img** |
| **ResNet-50 (Fine-Tuned)** | **99.63%** | **99.63%** | **99.64%** | **99.63%** | 24.03M | 91.9 MB | 1.23 ms/img |
| **Baseline CNN (Scratch)** | 68.15% | 68.53% | 74.55% | 68.15% | 0.36M | 1.4 MB | 0.14 ms/img |

> **Takeaway**: EfficientNet-B0 achieves the identical top-tier accuracy of ResNet-50 while reducing parameter count by **83.3%** and cutting inference latency by **55.3%**, making it the ideal candidate for edge deployment on NVIDIA Jetson or industrial PC units.

### Per-Class Performance (EfficientNet-B0 & ResNet-50)

```
                  precision    recall  f1-score   support
         crazing     1.0000    1.0000    1.0000        45
       inclusion     0.9783    1.0000    0.9890        45
         patches     1.0000    1.0000    1.0000        45
  pitted_surface     1.0000    1.0000    1.0000        45
 rolled_in_scale     1.0000    1.0000    1.0000        45
       scratches     1.0000    0.9778    0.9888        45

        accuracy                         0.9963       270
       macro avg     0.9964    0.9963    0.9963       270
    weighted avg     0.9964    0.9963    0.9963       270
```

---

## 🔍 Defect Taxonomy & Morphology

SteelSight recognizes all 6 defect categories from the Northeastern University (NEU) surface defect benchmark:

| Defect Class | Code | Visual Appearance | Metallurgical Cause | Production Impact |
| :--- | :---: | :--- | :--- | :--- |
| **Crazing** | `Cr` | Interconnected web of fine superficial cracks | Rapid differential cooling / thermal stress | **CRITICAL**: High fatigue propagation risk |
| **Inclusion** | `In` | Non-metallic dark spots or tail-like tails | Entrapped slag or deoxidation particles ($Al_2O_3$) | **MEDIUM**: Triggers stamping fractures |
| **Patches** | `Pa` | Irregular, plate-like darker oxidized layers | Uneven scale oxidation or reheating | **LOW**: Cosmetic; impairs coating adhesion |
| **Pitted Surface** | `PS` | Concentrated micro-craters and surface pits | Scale dislodgement / excessive pickling | **CRITICAL**: Severe wall thinning & corrosion |
| **Rolled-in Scale** | `RS` | Dark bands pressed directly into steel body | High-pressure descaling header malfunction | **MEDIUM**: Causes delamination under strain |
| **Scratches** | `Sc` | Directional, linear abrasion gouges | Friction against guide rails or conveyor rolls | **LOW/MED**: Stress risers if depth is high |

---

## 🏗️ System Architecture

```
                      +-----------------------------+
                      |   Industrial Camera Feed    |
                      |   (Line-scan / GigE Vision) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Preprocessing & CLAHE      |
                      |  - Contrast enhancement     |
                      |  - Albumentations pipeline  |
                      |  - Normalized tensor (3x224)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  EfficientNet-B0 Inference  |
                      |  - 4.02M params @ 0.55ms    |
                      |  - Logits & Softmax Probs   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Explainability Engine (XAI) |
                      | - Grad-CAM backpropagation  |
                      | - Jet-colormap overlay      |
                      | - Defect area coverage %    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Industrial Severity Engine  |
                      | - Risk Matrix Integration   |
                      | - Confidence & Area Weight  |
                      | - CRITICAL / MEDIUM / LOW   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    FastAPI Engine (Port 8000)|
                      +-------+---------------+-----+
                              |               |
               +--------------+               +---------------+
               v                                              v
+-------------------------------+              +-------------------------------+
|  SteelSight React HMI (Vite)  |              |   Streamlit Python Dashboard  |
|  - Live Image Inspection      |              |   - Checkpoint Switcher       |
|  - Model Benchmark Explorer   |              |   - Direct Grad-CAM Visualizer|
|  - Real-time Audit History    |              |   - Parameter Inspection      |
+-------------------------------+              +-------------------------------+
```

---

## 📁 Repository Structure

```
IndustryAI/
├── api/
│   └── server.py                 # FastAPI backend (Inference, Grad-CAM, Health check)
├── checkpoints/
│   ├── best_baseline.pth         # Trained weights: Baseline CNN (1.4 MB)
│   ├── best_efficientnet.pth     # Trained weights: EfficientNet-B0 (15.6 MB) [Deployed]
│   └── best_resnet50.pth         # Trained weights: ResNet-50 (91.9 MB)
├── data/
│   └── .gitkeep                  # Dataset root (data/NEU-CLS populated via downloader)
├── docs/
│   ├── ARCHITECTURE.md           # In-depth architectural & XAI specifications
│   ├── API.md                    # Complete REST API documentation & curl examples
│   ├── DATASET.md                # NEU benchmark guide, morphology & augmentation
│   └── DEPLOYMENT.md             # Edge computing, Docker, ONNX & TensorRT guide
├── outputs/
│   ├── confusion_matrix_resnet50.png
│   ├── model_comparison_benchmark.csv
│   ├── model_comparison_chart.png
│   ├── nb_gradcam_all_classes.png
│   ├── nb_training_curves_combined.png
│   ├── Industrial_Defect_Detection_Summary.pptx
│   └── Industrial_Defect_Detection_Viva_Guide.pdf
├── src/
│   ├── compare_models.py         # Automated benchmarking script (Latency, Params, F1)
│   ├── dataset.py                # Albumentations transforms & PyTorch Dataset
│   ├── eda.py                    # Exploratory data analysis scripts
│   ├── evaluate.py               # Evaluation loop, confusion matrix & metrics
│   ├── explainability.py         # Grad-CAM heatmap generation module
│   ├── models.py                 # ResNet-50, EfficientNet-B0, Baseline & Autoencoder
│   ├── predict.py                # CLI inference tool
│   └── train.py                  # Training loop with EarlyStopping & CosineAnnealing
├── steelsight/                   # React 19 + TypeScript + Vite + Tailwind CSS Dashboard
│   ├── app/
│   │   ├── components/layout/    # Navigation and UI shell
│   │   ├── lib/                  # API client, severity engine, state store
│   │   ├── routes/               # Inspect, Models, Analytics, History views
│   │   └── types/                # Inspection TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
├── app.py                        # Streamlit web application
├── download_data.py              # Automated NEU dataset downloader & extractor
├── requirements.txt              # Python dependencies
├── run_demo.bat                  # One-click launcher for Streamlit demo
├── start_steelsight.bat          # One-click launcher for FastAPI + React HMI
└── test_environment.py           # Environment and CUDA verification script
```

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python**: 3.10 or 3.11
- **Node.js**: v18+ & npm
- **CUDA** *(Optional)*: NVIDIA GPU with CUDA 12.1+ for accelerated training and real-time inference (CPU is also fully supported).

### 2. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/aakashbisht0704/steel-defect-detection.git
cd steel-defect-detection

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

Verify your Python environment and CUDA availability:
```bash
python test_environment.py
```

### 3. Install Frontend Dependencies

```bash
cd steelsight
npm install
cd ..
```

### 4. Download NEU Dataset

SteelSight provides an automated downloader that pulls and validates the NEU Surface Defect dataset:

```bash
python download_data.py
```

---

## 🚀 Running the Applications

### Option A: SteelSight Full-Stack Platform (FastAPI + React 19)

**Windows One-Click Launcher:**
```cmd
start_steelsight.bat
```

**Manual Startup:**
1. **Start the AI Engine (FastAPI)**:
   ```bash
   uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
   ```
   *API Swagger Docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs).*

2. **Start the SteelSight Frontend (Vite)**:
   ```bash
   cd steelsight
   npm run dev
   ```
   *HMI Dashboard will be live at [http://localhost:5173](http://localhost:5173).*

---

### Option B: Streamlit Inspection Lab

For rapid Python-centric testing and checkpoint inspection:

**Windows One-Click Launcher:**
```cmd
run_demo.bat
```

**Manual Startup:**
```bash
streamlit run app.py
```
*Streamlit dashboard will open at [http://localhost:8501](http://localhost:8501).*

---

## 🧪 Training & Evaluation CLI

Train and benchmark models directly via the modular scripts in `src/`:

### 1. Train a Model
```bash
# Train EfficientNet-B0 (Recommended)
python src/train.py --model efficientnet --epochs 25 --batch-size 32 --lr 1e-3

# Train ResNet-50 (Transfer learning with frozen backbone then fine-tuning)
python src/train.py --model resnet50 --epochs 30 --batch-size 32

# Train Baseline CNN from scratch
python src/train.py --model baseline --epochs 40 --batch-size 32
```

### 2. Evaluate a Model
```bash
python src/evaluate.py --model efficientnet --checkpoint checkpoints/best_efficientnet.pth
```

### 3. Generate Multi-Model Comparison Benchmarks
```bash
python src/compare_models.py
```
*Outputs latency, throughput, model sizes, and macro metrics to `outputs/model_comparison_benchmark.csv` and `outputs/model_comparison_chart.png`.*

### 4. Generate Grad-CAM Heatmaps
```bash
python src/explainability.py --image data/NEU-CLS/crazing/crazing_1.bmp --model efficientnet
```

---

## 📡 REST API Reference

### `POST /predict`
Uploads a surface image and returns predictions, confidence, attention coverage, and base64 Grad-CAM heatmap.

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@sample_defect.jpg"
```

**Response:**
```json
{
  "defect": "crazing",
  "confidence": 0.9984,
  "class_probabilities": {
    "crazing": 0.9984,
    "inclusion": 0.0003,
    "patches": 0.0002,
    "pitted_surface": 0.0005,
    "rolled_in_scale": 0.0004,
    "scratches": 0.0002
  },
  "attention_coverage": 0.142,
  "inference_ms": 0.52,
  "gradcam_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "original_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

*See [docs/API.md](docs/API.md) for full endpoint specifications, code examples, and error codes.*

---

## 📖 Further Documentation

- **[System Architecture & Deep Learning](docs/ARCHITECTURE.md)**: Network layers, transfer learning strategies, Grad-CAM mathematics, and severity matrix logic.
- **[REST API Reference](docs/API.md)**: OpenAPI schemas, curl, Python `requests`, and JavaScript `fetch` code samples.
- **[NEU Dataset Specification](docs/DATASET.md)**: Metallurgical defect origins, class distribution, and Albumentations augmentations.
- **[Production & Edge Deployment](docs/DEPLOYMENT.md)**: Docker containerization, NVIDIA Jetson edge inference, ONNX/TensorRT quantization, and PLC/SCADA integration.

---

## 🤝 Contributing

Contributions are welcomed! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for pull request guidelines, code formatting standards, and testing procedures.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Northeastern University (NEU), China**: For authoring and releasing the NEU Surface Defect Database benchmark.
- **PyTorch & torchvision**: For deep learning modeling primitives.
- **Albumentations**: For industrial image augmentation pipelines.
- **Jacob Gildenblat & pytorch-grad-cam**: For Grad-CAM implementation.
- **FastAPI & Vite/React Community**: For high-performance backend and frontend tooling.
