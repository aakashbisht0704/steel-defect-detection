"""
Streamlit Web Application for Industrial Defect Detection & Quality Inspection.
Run with:
    streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import get_val_transforms
from src.models import get_resnet50, get_efficientnet_b0, BaselineCNN

st.set_page_config(
    page_title="Industrial Defect Detection System",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Automated Industrial Surface Defect Inspection")
st.markdown("""
Real-time deep learning system for surface quality inspection on manufactured materials (NEU Surface Defect Benchmark).
Identifies defect categories, predicts confidence levels, and highlights defective surface regions via Grad-CAM explainability.
""")

# Sidebar settings
st.sidebar.header("⚙️ Model Configuration")
checkpoint_dir = "checkpoints"
available_checkpoints = []
if os.path.exists(checkpoint_dir):
    available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

if not available_checkpoints:
    st.sidebar.warning("⚠️ No model checkpoints found in `checkpoints/`. Train a model first via `python src/train.py`.")
    selected_ckpt = None
else:
    selected_ckpt = st.sidebar.selectbox("Select Model Checkpoint", available_checkpoints)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Supported Defect Categories:**
1. 🕸️ **Crazing (Cr)**: Micro-crack network
2. 🪨 **Inclusion (In)**: Foreign non-metallic particles
3. 🩹 **Patches (Pa)**: Oxidized irregular plates
4. 🕳️ **Pitted Surface (PS)**: Pits & craters
5. ⚖️ **Rolled-in Scale (RS)**: Embedded mill scale
6. ⚡ **Scratches (Sc)**: Directional abrasion gouges
""")

@st.cache_resource
def load_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "resnet50")
    idx_to_class = ckpt.get("idx_to_class", {})
    num_classes = len(idx_to_class) if idx_to_class else 6

    if model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, freeze_backbone=False)
        target_layer = [model.layer4[-1]]
    elif model_name == "efficientnet":
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=False)
        target_layer = [model.features[-1]]
    else:
        model = BaselineCNN(num_classes=num_classes)
        target_layer = [model.features[6]]

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, idx_to_class, target_layer, device

uploaded_file = st.file_uploader("Upload an Industrial Surface Image (.bmp, .jpg, .png)", type=["bmp", "jpg", "png", "jpeg"])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None and selected_ckpt is not None:
    ckpt_full_path = os.path.join(checkpoint_dir, selected_ckpt)
    model, idx_to_class, target_layer, device = load_model(ckpt_full_path)

    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_float = np.float32(img_resized) / 255.0

    transform = get_val_transforms(img_size=224)
    input_tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        conf, pred_idx = probs.max(0)

    pred_idx = pred_idx.item()
    pred_label = idx_to_class.get(pred_idx, f"Class {pred_idx}").capitalize()
    confidence = conf.item() * 100.0

    with col1:
        st.subheader("📷 Surface Image & Grad-CAM Heatmap")
        tab1, tab2, tab3 = st.tabs(["Original", "Grad-CAM Attention", "Defect Overlay"])
        
        with tab1:
            st.image(img_resized, caption="Input Surface Image", use_container_width=True)
            
        # Compute Grad-CAM
        cam = GradCAM(model=model, target_layers=target_layer)
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        with tab2:
            st.image(grayscale_cam, caption="Feature Activation Density", clamp=True, use_container_width=True)
        with tab3:
            st.image(cam_image, caption=f"Defect Localization: {pred_label}", use_container_width=True)

    with col2:
        st.subheader("🎯 Inspection Diagnosis")
        
        # Big metric card
        st.success(f"### Predicted Defect: **{pred_label}**")
        st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

        st.subheader("📊 Class Probability Distribution")
        probs_dict = {
            idx_to_class.get(i, f"Class_{i}").capitalize(): float(probs[i].item() * 100)
            for i in range(len(probs))
        }
        
        st.bar_chart(probs_dict)

        st.info("💡 **Quality Control Recommendation:** " + (
            "Surface requires manual rework / diverter ejection." if pred_label.lower() != "defect_free" else "Surface passes automated quality standards."
        ))
else:
    if uploaded_file is None:
        st.info("👉 Upload an industrial surface image above or select a sample image from the dataset.")
