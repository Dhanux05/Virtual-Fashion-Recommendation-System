import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import os
import base64

# Resolve paths relative to this file so the app works no matter what
# directory `streamlit run` is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------- Page config -------------------
st.set_page_config(page_title="Fashion Recommendation System", layout="wide")

# ------------------- Background + overlay -------------------
def set_background(img_file):
    # NOTE: "background.jpg" in this project is a *folder* of candidate
    # background images (b1.png, b2.webp, b3.jpg, ...), not an image file.
    # A bare `url("background.jpg")` pointing at a directory can never
    # render, so the caller now passes a real file inside it.
    if os.path.exists(img_file) and os.path.isfile(img_file):
        with open(img_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(img_file)[1].lstrip(".") or "jpg"
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image:
                linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)),
                url("data:image/{ext};base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}

            .title-box {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.4);
                margin-bottom: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"⚠ background image not found: {img_file}")

set_background(os.path.join(BASE_DIR, "background.jpg", "b3.jpg"))

# ------------------- Title -------------------
st.markdown("""
<div class="title-box">
    <h1 style="color:black;">Fashion Recommendation System</h1>
</div>
""", unsafe_allow_html=True)

# ------------------- Load data -------------------
image_features = pickle.load(open(os.path.join(BASE_DIR, "image_features_embedding.pkl"), "rb"))
img_files = pickle.load(open(os.path.join(BASE_DIR, "img_files.pkl"), "rb"))
image_features = np.array(image_features)

# ------------------- Load model -------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    model = Sequential([base_model, GlobalMaxPooling2D()])
    return model

model = load_model()

# ------------------- Feature extraction -------------------
def extract_features(uploaded_file, model):
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0).flatten()
    return features / norm(features)

# ------------------- Recommendation -------------------
def recommend_images(features, image_features, img_files, top_n=5):
    scores = np.dot(image_features, features)
    idx = np.argsort(scores)[::-1][:top_n]
    return [(img_files[i], scores[i]) for i in idx]


def get_image_path(img_path_from_list):
    """Resolve a path recorded in img_files.pkl (e.g. 'Img_Dataset\\10000.jpg')
    to a real file next to this script, regardless of OS path separators or
    the process's current working directory."""
    filename = str(img_path_from_list).replace("\\", "/").split("/")[-1]
    return os.path.join(BASE_DIR, "Img_Dataset", filename)

# ------------------- Upload -------------------
uploaded_file = st.file_uploader("📤 Upload a fashion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    features = extract_features(uploaded_file, model)

    st.markdown("<h3 style='color:white;'>Uploaded Image</h3>", unsafe_allow_html=True)
    st.image(uploaded_file, width=300)

    st.markdown("<h3 style='color:white;'>Top Recommended Images</h3>", unsafe_allow_html=True)

    recommendations = recommend_images(features, image_features, img_files)

    cols = st.columns(5)
    for i, (img_path, score) in enumerate(recommendations):
        with cols[i]:
            resolved_path = get_image_path(img_path)
            if os.path.exists(resolved_path):
                st.image(resolved_path, use_container_width=True)
            else:
                st.error(f"Image not found: {os.path.basename(resolved_path)}")
            st.caption(f"Similarity: {score:.4f}")
