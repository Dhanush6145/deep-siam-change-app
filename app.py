import streamlit as st
import torch
import os
import gdown
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    download_model()

    model = SiameseChangeNet()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

# -------------------------
# TRANSFORM (IMPORTANT FIX)
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
# PREDICTION
# -------------------------
def predict(model, img1, img2):
    t1 = transform(img1).unsqueeze(0).to(DEVICE)
    t2 = transform(img2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(t1, t2)
        output = torch.sigmoid(output)

    output = output.squeeze().cpu().numpy()

    # DEBUG VALUES
    min_val, max_val = output.min(), output.max()

    # LOWER THRESHOLD (important)
    binary = (output > 0.3).astype("float32")

    return output, binary, min_val, max_val

# -------------------------
# UI
# -------------------------
st.title("🧠 Deep Siam Change Detection")

model = load_model()

file1 = st.file_uploader("Upload Image 1", type=["jpg", "png"])
file2 = st.file_uploader("Upload Image 2", type=["jpg", "png"])

if file1 and file2:
    img1 = Image.open(file1).convert("RGB")
    img2 = Image.open(file2).convert("RGB")

    st.image([img1, img2], caption=["Image 1", "Image 2"])

    if st.button("Predict Change"):
        raw, binary, min_val, max_val = predict(model, img1, img2)

        st.write(f"Min: {min_val:.4f}, Max: {max_val:.4f}")

        # HEATMAP
        st.subheader("🔥 Heatmap")
        fig, ax = plt.subplots()
        ax.imshow(raw, cmap="jet")
        ax.axis("off")
        st.pyplot(fig)

        # BINARY MAP
        st.subheader("✅ Change Map")
        st.image(binary, clamp=True)
