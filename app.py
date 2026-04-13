import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import requests
import importlib

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "best_model.pth"
FILE_ID = "1g7WJ48be5zoAXTBCdcB1Ra6fYNom3I5r"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# DOWNLOAD MODEL
# ---------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model...")

        try:
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)

            st.success("✅ Model downloaded!")

        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            st.stop()

# ---------------------------
# AUTO LOAD MODEL CLASS
# ---------------------------
def get_model_class():
    try:
        model_module = importlib.import_module("models.model")

        # Try common names
        possible_names = [
            "SiameseNetwork",
            "SiameseNet",
            "ChangeDetectionModel",
            "Model",
            "Net"
        ]

        for name in possible_names:
            if hasattr(model_module, name):
                return getattr(model_module, name)

        st.error("❌ No valid model class found in models/model.py")
        st.stop()

    except Exception as e:
        st.error(f"❌ Import error: {e}")
        st.stop()

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    download_model()

    ModelClass = get_model_class()

    try:
        model = ModelClass()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# ---------------------------
# IMAGE TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict(model, img1, img2):
    t1 = transform(img1).unsqueeze(0).to(DEVICE)
    t2 = transform(img2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        try:
            output = model(t1, t2)  # Siamese input
        except:
            # fallback if model takes single input
            output = model(t1)

    return output.cpu().numpy()

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Deep Siam Change Detection", layout="centered")

st.title("🧠 Deep Siam Change Detection")

model = load_model()

# Upload two images
file1 = st.file_uploader("📤 Upload Image 1", type=["jpg", "png", "jpeg"])
file2 = st.file_uploader("📤 Upload Image 2", type=["jpg", "png", "jpeg"])

if file1 and file2:
    img1 = Image.open(file1).convert("RGB")
    img2 = Image.open(file2).convert("RGB")

    st.image([img1, img2], caption=["Image 1", "Image 2"], use_container_width=True)

    if st.button("🔍 Predict"):
        result = predict(model, img1, img2)

        st.write("### ✅ Prediction Output:")
        st.write(result)
