import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import gdown

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "best_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1g7WJ48be5zoAXTBCdcB1Ra6fYNom3I5r"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

    # IMPORT YOUR MODEL
    from models.model import SiameseNetwork  # adjust if different

    model = SiameseNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# -------------------------------
# IMAGE PREPROCESS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocess(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(img1, img2):
    t1 = preprocess(img1)
    t2 = preprocess(img2)

    with torch.no_grad():
        output = model(t1, t2)

        # Adjust based on your model output
        output = torch.sigmoid(output)

        output = output.squeeze().cpu().numpy()

    return output

# -------------------------------
# UI
# -------------------------------
st.title("🔍 Change Detection App (Deep Siamese Model)")

st.write("Upload two images to detect changes")

col1, col2 = st.columns(2)

with col1:
    img1_file = st.file_uploader("Upload Image 1 (Before)", type=["png", "jpg", "jpeg"])

with col2:
    img2_file = st.file_uploader("Upload Image 2 (After)", type=["png", "jpg", "jpeg"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    st.subheader("Input Images")
    st.image([img1, img2], caption=["Before", "After"], width=300)

    if st.button("Run Prediction"):
        result = predict(img1, img2)

        st.subheader("Prediction Output")

        # Normalize for display
        result_img = (result - result.min()) / (result.max() - result.min() + 1e-8)

        st.image(result_img, caption="Detected Changes", use_column_width=True)
