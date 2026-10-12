# SteelSight Production & Edge Deployment Guide

This guide covers deploying SteelSight in manufacturing and factory edge environments.

---

## 🏭 Deployment Scenarios

```
                       Industrial Environment
                       ----------------------
               [ GigE Vision / Basler Camera ]
                              │
                    RTSP / Video Stream
                              │
                              ▼
                 +──────────────────────────+
                 │   Edge Computing Device  │
                 │ (NVIDIA Jetson / x86 PC) │
                 │                          │
                 │  - SteelSight AI Engine  │
                 │  - EfficientNet-B0 ONNX  │
                 │  - FastAPI Daemon        │
                 +─────────────┬────────────+
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   [ SteelSight React HMI ]         [ Industrial PLC / SCADA ]
   (Operator Touchscreen)           (Modbus / MQTT / OPC-UA)
                                    - Automated Diverter Arm
                                    - Sound Alarms on Critical
```

---

## 🐳 Docker Deployment

### 1. `Dockerfile` (AI Engine Backend)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY checkpoints/ checkpoints/

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  ai-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CHECKPOINT_PATH=/app/checkpoints/best_efficientnet.pth
    restart: always

  frontend:
    build:
      context: ./steelsight
    ports:
      - "5173:80"
    depends_on:
      - ai-engine
    restart: always
```

---

## ⚡ Edge Acceleration (NVIDIA Jetson & ONNX)

In high-speed rolling mills (running at up to 10–20 meters/second), low inference latency is paramount. SteelSight's EfficientNet-B0 can be exported to ONNX and optimized via TensorRT:

### Exporting to ONNX
```python
import torch
from torchvision import models

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 6)
model.load_state_dict(torch.load("checkpoints/best_efficientnet.pth")["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, 
    dummy_input, 
    "checkpoints/efficientnet_b0.onnx", 
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("Model successfully exported to ONNX!")
```

### Compiling with TensorRT
```bash
trtexec --onnx=checkpoints/efficientnet_b0.onnx \
        --saveEngine=checkpoints/efficientnet_b0.engine \
        --fp16
```

Expected TensorRT latency on an NVIDIA Jetson Orin Nano is **$< 0.35\text{ ms}$ per frame**, exceeding standard 60 FPS line-scan camera requirements.

---

## 📡 PLC & SCADA Integration

When the SteelSight API returns a defect flagged as `CRITICAL` (e.g. Crazing or Pitted Surface):

1. **REST Webhook / MQTT**:
   SteelSight can publish MQTT events to a broker connected to the plant PLC:
   ```json
   {
     "topic": "factory/line1/inspection",
     "payload": {
       "defect": "crazing",
       "severity": "CRITICAL",
       "confidence": 0.998,
       "timestamp": "2026-10-12T10:00:00Z"
     }
   }
   ```
2. **Pneumatic Diverter**:
   PLC triggers a pneumatic diverter gate to reject the defective steel sheet into the scrap bin, preventing flawed material from reaching customer coils.
