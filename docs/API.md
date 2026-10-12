# SteelSight REST API Documentation

The SteelSight AI Engine backend is powered by **FastAPI**. It delivers high-throughput, low-latency inference, real-time Grad-CAM heatmap rendering, and hardware health telemetry.

- **Base URL**: `http://localhost:8000`
- **Interactive OpenAPI Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 📡 Endpoints

### 1. Health Check
Checks the status of the inference service and active model.

- **Method**: `GET`
- **Endpoint**: `/health`
- **Response**:
  ```json
  {
    "status": "online",
    "model": "EfficientNet-B0",
    "version": "1.0"
  }
  ```

---

### 2. Model Information & Benchmarks
Returns deployment specifications and comparative benchmarks across all tested models.

- **Method**: `GET`
- **Endpoint**: `/model-info`
- **Response**:
  ```json
  {
    "deployed": {
      "name": "EfficientNet-B0",
      "accuracy": 99.63,
      "macro_f1": 99.63,
      "parameters": "4.02M",
      "inference_ms": 0.55
    },
    "benchmarks": [
      {
        "name": "Baseline CNN",
        "accuracy": 68.15,
        "macro_f1": 65.20,
        "parameters": "1.2M",
        "inference_ms": 0.40
      },
      {
        "name": "ResNet-50",
        "accuracy": 99.63,
        "macro_f1": 99.60,
        "parameters": "23.5M",
        "inference_ms": 1.20
      },
      {
        "name": "EfficientNet-B0",
        "accuracy": 99.63,
        "macro_f1": 99.63,
        "parameters": "4.02M",
        "inference_ms": 0.55
      }
    ]
  }
  ```

---

### 3. Defect Prediction & Grad-CAM Analysis
Accepts an image file, runs deep learning classification, generates a Grad-CAM saliency map, calculates surface coverage, and outputs structured inspection results.

- **Method**: `POST`
- **Endpoint**: `/predict`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` (binary file, required): Surface image (`.bmp`, `.jpg`, `.jpeg`, `.png`).

- **Response**:
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

#### Response Fields:
| Field | Type | Description |
| :--- | :--- | :--- |
| `defect` | string | Top predicted defect category (`crazing`, `inclusion`, etc.) |
| `confidence` | float | Softmax probability for the predicted class ($[0.0, 1.0]$) |
| `class_probabilities`| object | Normalized probability distribution across all 6 defect categories |
| `attention_coverage` | float | Fraction of the image surface activated by Grad-CAM ($> 0.5$ intensity) |
| `inference_ms` | float | Forward pass computation duration in milliseconds |
| `gradcam_image` | string | Jet-colormap Grad-CAM overlay encoded as a base64 Data URL |
| `original_image` | string | Resized input image encoded as a base64 Data URL |

---

## 💻 Code Examples

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@data/NEU-CLS/crazing/crazing_1.bmp"
```

### Python (`requests`)
```python
import requests

url = "http://localhost:8000/predict"
file_path = "sample_defect.jpg"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"image": f})

data = response.json()
print(f"Detected Defect: {data['defect']}")
print(f"Confidence:      {data['confidence'] * 100:.2f}%")
print(f"Latency:         {data['inference_ms']:.2f} ms")
print(f"Defect Coverage: {data['attention_coverage'] * 100:.1f}%")
```

### JavaScript / TypeScript (`fetch`)
```typescript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log(`Defect: ${result.defect}, Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

---

## 🔒 Error Handling

| Status Code | Reason | Resolution |
| :--- | :--- | :--- |
| `400 Bad Request` | Unreadable image format or missing file | Ensure payload is sent as `multipart/form-data` with key `image` |
| `500 Internal Error` | Missing checkpoint or CUDA device failure | Verify checkpoint file exists in `checkpoints/` |
| `503 Service Unavailable` | Model initialization in progress | Wait for server startup |
