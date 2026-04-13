import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import requests

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "best_model.pth"

# 🔥 Replace with your file ID
FILE_ID = "1g7WJ48be5zoAXTBCdcB1Ra6fYNom3I5r"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# DOWNLOAD MODEL (SAFE WAY)
# ---------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model... (first time only)")

        try:
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)

            st.success("✅ Model downloaded successfully!")

        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            st.stop()

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    download_model()

    try:
        from models.model import SiameseNetwork  # your model

        model = SiameseNetwork()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# ---------------------------
# IMAGE PREPROCESS
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict(model, img):
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img, img)  # adjust if needed

    return output.cpu().numpy()

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Deep Siam Change Detection")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        result = predict(model, image)
        st.write("### ✅ Prediction Output:")
        st.write(result)
