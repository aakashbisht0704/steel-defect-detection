import base64
import time
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define constants & paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints" / "best_efficientnet.pth"))

app = FastAPI(title="SteelSight API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, len(CLASSES))
)

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    print("Model checkpoint loaded.")
except Exception as e:
    print(f"Warning: Could not load checkpoint from {CHECKPOINT_PATH}: {e}")

model = model.to(device)
model.eval()

# Grad-CAM target layer
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

@app.get("/health")
def health_check():
    return {"status": "online", "model": "EfficientNet-B0", "version": "1.0"}

@app.get("/model-info")
def model_info():
    return {
        "deployed": { "name": "EfficientNet-B0", "accuracy": 99.63, "macro_f1": 99.63, "parameters": "4.02M", "inference_ms": 0.55 },
        "benchmarks": [
            { "name": "Baseline CNN", "accuracy": 68.15, "macro_f1": 65.20, "parameters": "1.2M", "inference_ms": 0.40 },
            { "name": "ResNet-50", "accuracy": 99.63, "macro_f1": 99.60, "parameters": "23.5M", "inference_ms": 1.20 },
            { "name": "EfficientNet-B0", "accuracy": 99.63, "macro_f1": 99.63, "parameters": "4.02M", "inference_ms": 0.55 }
        ]
    }

def encode_image(img_array):
    success, buffer = cv2.imencode('.png', img_array)
    if not success:
        return ""
    img_bytes = buffer.tobytes()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    orig_img_resized = cv2.resize(img, (224, 224))
    orig_img_resized_rgb = orig_img_resized.copy() # keep rgb for output
    
    # Transform
    transformed = transform(image=img)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    inference_ms = (time.time() - start_time) * 1000
    
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    attention_coverage = float(np.sum(grayscale_cam > 0.5) / grayscale_cam.size)
    
    # Create CAM image 
    # show_cam_on_image needs float32 image in [0,1]
    img_float = orig_img_resized_rgb.astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    # Generate heatmap only
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return {
        "prediction": pred_class,
        "confidence": confidence,
        "probabilities": {cls: float(prob) for cls, prob in zip(CLASSES, probs)},
        "gradcam": encode_image(cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)),
        "gradcam_overlay": encode_image(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)),
        "original_image": encode_image(cv2.cvtColor(orig_img_resized_rgb, cv2.COLOR_RGB2BGR)),
        "attention_coverage": attention_coverage,
        "model": {
            "name": "EfficientNet-B0",
            "version": "1.0",
            "accuracy": 99.63,
            "macro_f1": 99.63,
            "parameters": "4.02M",
            "inference_ms": round(inference_ms, 2)
        }
    }
