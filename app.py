import streamlit as st
import torch
import os
import gdown
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ✅ Import your correct model class
from models.model import SiameseChangeNet

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model.pth"
FILE_ID = "1g7WJ48be5zoAXTBCdcB1Ra6fYNom3I5r"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# DOWNLOAD MODEL
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------
# LOAD MODEL (ROBUST)
# -------------------------
@st.cache_resource
def load_model():
    download_model()

    try:
        # ✅ Load as state_dict
        model = SiameseChangeNet()
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model

    except Exception:
        st.warning("⚠️ Trying fallback model loading...")

        try:
            model = torch.load(MODEL_PATH, map_location=DEVICE)
            model.eval()
            return model
        except Exception as e:
            st.error(f"❌ Model loading failed:\n{e}")
            st.stop()

# -------------------------
# IMAGE TRANSFORM (FIXED)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# PREDICT FUNCTION (FIXED)
# -------------------------
def predict(model, img1, img2):
    t1 = transform(img1).unsqueeze(0).to(DEVICE)
    t2 = transform(img2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(t1, t2)
        output = torch.sigmoid(output)

    output = output.squeeze().cpu().numpy()

    # ✅ Debug values
    min_val, max_val = output.min(), output.max()

    # ✅ Binary threshold
    binary = (output > 0.5).astype("float32")

    return output, binary, min_val, max_val

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Deep Siam Change Detection")

st.title("🧠 Deep Siam Change Detection")

model = load_model()

# Upload images
file1 = st.file_uploader("📤 Upload Image 1", type=["jpg", "png", "jpeg"])
file2 = st.file_uploader("📤 Upload Image 2", type=["jpg", "png", "jpeg"])

if file1 and file2:
    img1 = Image.open(file1).convert("RGB")
    img2 = Image.open(file2).convert("RGB")

    st.image([img1, img2], caption=["Image 1", "Image 2"], use_container_width=True)

    if st.button("🔍 Predict Change"):

        raw, binary, min_val, max_val = predict(model, img1, img2)

        st.subheader("📊 Debug Info")
        st.write(f"Min Value: {min_val:.4f}")
        st.write(f"Max Value: {max_val:.4f}")

        # -------------------------
        # OUTPUTS
        # -------------------------

        st.subheader("🔥 Heatmap (Raw Output)")
        st.image(raw, use_container_width=True)

        st.subheader("✅ Binary Change Map")
        st.image(binary * 255, use_container_width=True)
