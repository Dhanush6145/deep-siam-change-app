import streamlit as st
import torch
import gdown
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Import YOUR model correctly
from models.model import SiameseChangeNet

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "best_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1g7WJ48be5zoAXTBCdcB1Ra6fYNom3I5r"

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = SiameseChangeNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    return model

model = load_model()

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# -------------------------
# UI
# -------------------------
st.title("🧠 Deep Siam Change Detection")

st.write("Upload two images to detect changes")

img1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
img2 = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])

# -------------------------
# PREDICTION
# -------------------------
if img1 and img2:
    image1 = Image.open(img1).convert("RGB")
    image2 = Image.open(img2).convert("RGB")

    st.image([image1, image2], caption=["Image 1", "Image 2"], width=300)

    x1 = transform(image1).unsqueeze(0)
    x2 = transform(image2).unsqueeze(0)

    with torch.no_grad():
        output = model(x1, x2)
        output = torch.sigmoid(output)
        output = output.squeeze().numpy()

    st.subheader("🔍 Change Map")
    st.image(output, clamp=True)
