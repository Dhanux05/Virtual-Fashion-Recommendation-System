import streamlit as st
import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D

# ------------------------------
# Load stored features
# ------------------------------
image_features = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files = pickle.load(open("img_files.pkl", "rb"))

# ------------------------------
# Model for feature extraction
# ------------------------------
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

def extract_features(uploaded_image, model):
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    processed = preprocess_input(expand_img)
    result = model.predict(processed).flatten()
    return result / norm(result)

def recommend(features, image_features):
    distances = np.linalg.norm(image_features - features, axis=1)
    index_list = np.argsort(distances)[:6]   # top 6 images
    return index_list

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("👗 Fashion Recommendation System")
st.write("Upload a clothing image to find similar outfits!")

uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_img.getbuffer())

    features = extract_features("temp.jpg", model)
    indices = recommend(features, np.array(image_features))

    st.subheader("Recommended Images")

    cols = st.columns(3)
    i = 0
    for idx in indices:
        with cols[i % 3]:
            st.image(img_files[idx])
        i += 1
