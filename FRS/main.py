import streamlit as st
import tensorflow
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

# Load pickle files with proper path handling
# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try multiple possible paths for pickle files
pkl_paths = [
    os.path.join(BASE_DIR, "image_features_embedding.pkl"),
    os.path.join(BASE_DIR, "..", "image_features_embedding.pkl"),
    "image_features_embedding.pkl",
    "FRS/image_features_embedding.pkl"
]

img_paths = [
    os.path.join(BASE_DIR, "img_files.pkl"),
    os.path.join(BASE_DIR, "..", "img_files.pkl"),
    "img_files.pkl",
    "FRS/img_files.pkl"
]

# Find and load features_list
features_list = None
features_path = None
for path in pkl_paths:
    if os.path.exists(path):
        try:
            features_list = pickle.load(open(path, "rb"))
            features_path = path
            break
        except Exception as e:
            continue

# Find and load img_files_list
img_files_list = None
img_files_path = None
for path in img_paths:
    if os.path.exists(path):
        try:
            img_files_list = pickle.load(open(path, "rb"))
            img_files_path = path
            break
        except Exception as e:
            continue

# Check if files were loaded successfully
if features_list is None or img_files_list is None:
    st.error("⚠️ Required model files (.pkl) not found!")
    st.warning("""
    **Missing Files:**
    - `image_features_embedding.pkl`
    - `img_files.pkl`
    
    These files are required for the recommendation system to work.
    Please ensure they are in the FRS directory.
    
    **Searched paths:**
    - FRS/image_features_embedding.pkl
    - FRS/img_files.pkl
    - Current directory
    """)
    st.stop()

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Fashion Animated Background
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;500;600;700&family=Great+Vibes&family=Pacifico&family=Satisfy&family=Poppins:wght@300;400;600&display=swap');
    
    /* Override Streamlit default background */
    html, body, [class*="css"] {
        background: transparent !important;
    }
    
    /* Fashion animated gradient background - modern dark theme */
    .stApp {
        background: linear-gradient(-45deg, #0a0e27, #1a1f3a, #2d1b4e, #1e3a5f, #0f4c75, #0a0e27) !important;
        background-size: 400% 400% !important;
        animation: gradient 20s ease infinite !important;
        position: relative;
        overflow: hidden;
        min-height: 100vh;
    }
    
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Fashion pattern overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text x="10" y="50" font-size="40" opacity="0.12">👗</text><text x="60" y="80" font-size="40" opacity="0.12">👠</text></svg>'),
            radial-gradient(circle at 20% 30%, rgba(45, 27, 78, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(30, 58, 95, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(15, 76, 117, 0.25) 0%, transparent 60%),
            radial-gradient(circle at 30% 70%, rgba(74, 144, 226, 0.15) 0%, transparent 50%);
        background-size: 200px 200px, 100% 100%, 100% 100%, 100% 100%;
        animation: fashionPattern 25s linear infinite, pulse 10s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes fashionPattern {
        0% {
            background-position: 0 0, 0 0, 0 0, 0 0;
        }
        100% {
            background-position: 200px 200px, 0 0, 0 0, 0 0;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 0.4;
            transform: scale(1);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.05);
        }
    }
    
    /* Floating fashion icons - moving everywhere across screen */
    .stApp::after {
        content: '👗 👠 👕 👔 👜 👒 💄 👑 👗 👠 👕 👔 👜 👒 💄 👑 👗 👠 👕 👔 👜 👒 💄 👑 👗 👠 👕 👔 👜 👒 💄 👑';
        font-size: 3.5rem;
        opacity: 0.12;
        position: fixed;
        top: 0;
        left: 0;
        width: 300%;
        height: 300%;
        pointer-events: none;
        z-index: 0;
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        grid-template-rows: repeat(12, 1fr);
        place-items: center;
        animation: floatEverywhere 50s linear infinite;
        will-change: transform;
    }
    
    @keyframes floatEverywhere {
        0% {
            transform: translate(0, 0) rotate(0deg);
        }
        20% {
            transform: translate(-20vw, -15vh) rotate(72deg);
        }
        40% {
            transform: translate(-40vw, 10vh) rotate(144deg);
        }
        60% {
            transform: translate(-30vw, 40vh) rotate(216deg);
        }
        80% {
            transform: translate(-10vw, 30vh) rotate(288deg);
        }
        100% {
            transform: translate(0, 0) rotate(360deg);
        }
    }
    
    /* Main content styling - dark theme, no white background */
    .main .block-container {
        background: transparent !important;
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: none !important;
        backdrop-filter: none;
        border: none !important;
        position: relative;
        z-index: 1;
        margin-top: 2rem;
    }
    
    /* Title styling - Animated and Attractive - Script Font */
    h1 {
        font-family: 'Dancing Script', 'Great Vibes', 'Pacifico', cursive !important;
        text-align: center;
        font-size: 4.5rem;
        margin-bottom: 2rem;
        font-weight: 700;
        position: relative;
        z-index: 2;
        letter-spacing: 2px;
        text-transform: none;
        font-style: normal;
        background: linear-gradient(135deg, #00d4ff 0%, #5b9fff 25%, #a855f7 50%, #ec4899 75%, #00d4ff 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease infinite, titleGlow 2s ease-in-out infinite alternate, titleFloat 4s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5), 0 0 60px rgba(168, 85, 247, 0.4);
        filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.6));
    }
    
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    @keyframes titleGlow {
        0% {
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5), 0 0 40px rgba(91, 159, 255, 0.4), 0 0 60px rgba(168, 85, 247, 0.3);
            filter: drop-shadow(0 0 8px rgba(0, 212, 255, 0.5));
        }
        100% {
            text-shadow: 0 0 40px rgba(0, 212, 255, 0.8), 0 0 80px rgba(91, 159, 255, 0.6), 0 0 120px rgba(236, 72, 153, 0.5);
            filter: drop-shadow(0 0 15px rgba(168, 85, 247, 0.8));
        }
    }
    
    @keyframes titleFloat {
        0%, 100% {
            transform: translateY(0px) scale(1);
        }
        50% {
            transform: translateY(-8px) scale(1.02);
        }
    }
    
    /* Sparkle effect around title */
    h1::before {
        content: '✨';
        position: absolute;
        left: -40px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2rem;
        animation: sparkleLeft 2s ease-in-out infinite;
        opacity: 0.8;
    }
    
    h1::after {
        content: '✨';
        position: absolute;
        right: -40px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2rem;
        animation: sparkleRight 2s ease-in-out infinite;
        opacity: 0.8;
    }
    
    @keyframes sparkleLeft {
        0%, 100% {
            transform: translateY(-50%) scale(1) rotate(0deg);
            opacity: 0.6;
        }
        50% {
            transform: translateY(-50%) scale(1.3) rotate(180deg);
            opacity: 1;
        }
    }
    
    @keyframes sparkleRight {
        0%, 100% {
            transform: translateY(-50%) scale(1) rotate(0deg);
            opacity: 0.6;
        }
        50% {
            transform: translateY(-50%) scale(1.3) rotate(-180deg);
            opacity: 1;
        }
    }
    
    /* All text elements - ensure visibility */
    p, label, div, span, [class*="st"] {
        color: #ffffff !important;
        position: relative;
        z-index: 2;
    }
    
    /* Text in main container - white for dark background */
    .main .block-container p,
    .main .block-container label,
    .main .block-container div,
    .main .block-container span {
        color: #ffffff !important;
    }
    
    /* File uploader label on dark background - make it white */
    .stFileUploader > label,
    label[data-testid="stFileUploaderLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* Text inside file uploader - make it white for dark background */
    .stFileUploader > div p,
    .stFileUploader > div div,
    .stFileUploader > div span,
    .stFileUploader > div small {
        color: #ffffff !important;
    }
    
    /* File uploader styling - dark theme, no white background */
    .stFileUploader > div {
        background: rgba(26, 26, 46, 0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        position: relative;
        z-index: 2;
        border: 3px dashed rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(102, 126, 234, 0.7);
        background: rgba(26, 26, 46, 0.85) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Inner drag and drop box - dark theme */
    .stFileUploader > div > div,
    [data-testid="stFileUploader"] > div > div,
    .stFileUploader [role="button"],
    .stFileUploader [class*="upload"],
    .stFileUploader [class*="drop"] {
        background: rgba(30, 30, 50, 0.8) !important;
        background-color: rgba(30, 30, 50, 0.8) !important;
        color: #ffffff !important;
    }
    
    /* All nested divs in file uploader - dark theme */
    .stFileUploader div div,
    .stFileUploader div div div {
        background: rgba(30, 30, 50, 0.8) !important;
        color: #ffffff !important;
    }
    
    /* File uploader text - ensure visibility */
    .stFileUploader label,
    .stFileUploader > label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* Text inside the drag and drop area - white for dark background */
    .stFileUploader p,
    .stFileUploader div,
    .stFileUploader span,
    .stFileUploader small,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Inner file uploader box - dark theme */
    .stFileUploader > div > div,
    [data-testid="stFileUploader"] > div > div,
    .stFileUploader [class*="uploadedFile"],
    .stFileUploader [class*="fileUploader"] {
        background: rgba(30, 30, 50, 0.8) !important;
        color: #ffffff !important;
    }
    
    /* All text inside file uploader container - white */
    [data-testid="stFileUploader"] * {
        color: #ffffff !important;
    }
    
    /* Streamlit file uploader text elements - white */
    .stFileUploader > div > p,
    .stFileUploader > div > span,
    .stFileUploader > div > div > p,
    .stFileUploader > div > div > span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Specific text in upload area - white */
    .stFileUploader [class*="upload"],
    .stFileUploader [class*="drop"],
    .stFileUploader [class*="file"] {
        color: #ffffff !important;
    }
    
    /* Image containers - ensure visibility - FORCE VISIBLE */
    [data-testid="stImage"],
    img,
    .stImage img,
    .stImage,
    [data-testid="stImage"] img,
    img[src],
    img[alt],
    .stImage > img,
    div[data-testid="stImage"] img {
        border-radius: 15px !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        position: relative !important;
        z-index: 999 !important;
        background: rgba(255, 255, 255, 0.1) !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        max-width: 100% !important;
        width: 100% !important;
        min-width: 100px !important;
        height: auto !important;
        min-height: 100px !important;
        object-fit: contain !important;
    }
    
    [data-testid="stImage"]:hover,
    img:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* Ensure image columns are visible */
    [data-testid="column"] img,
    .stColumn img,
    [data-testid="column"] [data-testid="stImage"],
    [data-testid="column"] .stImage {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
    }
    
    /* Remove any hiding styles */
    img[style*="display: none"],
    img[style*="visibility: hidden"],
    img[style*="opacity: 0"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Column headers - ensure visibility */
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #2c3e50 !important;
        text-align: center;
        position: relative;
        z-index: 2;
        font-weight: 600 !important;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.8);
        background: rgba(255, 255, 255, 0.9);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        display: inline-block;
    }
    
    /* Button styling - attractive gradient */
    button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        z-index: 2;
        position: relative;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 50%, #667eea 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* File uploader button specific styling */
    .stFileUploader button,
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader button:hover,
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 50%, #667eea 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Ensure all Streamlit elements have proper z-index and visibility */
    section[data-testid="stSidebar"],
    .stSidebar {
        background: rgba(255, 255, 255, 0.98) !important;
        z-index: 10;
    }
    
    /* Text in all containers */
    .element-container,
    .stMarkdown,
    .stText {
        color: #ffffff !important;
        z-index: 2;
        position: relative;
    }
    
    /* History column styling */
    [data-testid="column"]:last-child {
        background: rgba(26, 26, 46, 0.5) !important;
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    /* History expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 50, 0.8) !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .streamlit-expanderContent {
        background: rgba(20, 20, 40, 0.9) !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* History button styling */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title('✨ SnapStyle ✨')
st.markdown("""
<div style='text-align: center; margin-top: -20px; margin-bottom: 30px;'>
    <h2 style='color: #ffffff; font-family: "Playfair Display", serif; font-weight: 400; 
                letter-spacing: 3px; font-size: 1.8rem; margin: 0;
                background: linear-gradient(135deg, #00d4ff 0%, #5b9fff 50%, #a855f7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
                animation: subtitleGlow 3s ease-in-out infinite alternate;'>
        Virtual Fashion Recommendation System
    </h2>
</div>
<style>
    @keyframes subtitleGlow {
        0% {
            filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5));
        }
        100% {
            filter: drop-shadow(0 0 20px rgba(168, 85, 247, 0.8));
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []


def save_file(uploaded_file):
    try:
        # Get the base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        uploader_dir = os.path.join(base_dir, "uploader")
        
        # Create uploader directory if it doesn't exist
        if not os.path.exists(uploader_dir):
            os.makedirs(uploader_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(uploader_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)

    return result_normlized


def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices


def get_image_path(img_path_from_list):
    """
    Resolve image path from img_files_list to actual file location.
    Tries multiple possible paths to find the image.
    """
    # Get base directory (FRS directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Normalize the path - handle both forward and backward slashes
    # Extract just the filename from the path (handles paths like "Img_Dataset\10432.jpg")
    normalized_path = os.path.normpath(str(img_path_from_list))
    
    # Extract filename - if path contains "Img_Dataset", extract just the filename part
    # This prevents duplication when joining paths
    path_str = str(img_path_from_list).replace('\\', '/')  # Normalize to forward slashes first
    path_parts = path_str.split('/')
    
    # If path starts with "Img_Dataset", extract just the filename
    if len(path_parts) > 1 and path_parts[0].lower() == "img_dataset":
        filename = path_parts[-1]  # Get just the filename
    else:
        # Otherwise, use basename
        filename = os.path.basename(normalized_path)
    
    # Possible paths to try (prioritize most likely locations)
    # Always use just the filename to avoid duplication
    possible_paths = [
        # Most likely: Img_Dataset folder in FRS directory (base_dir is FRS)
        os.path.join(base_dir, "Img_Dataset", filename),
        # Try with absolute path
        os.path.abspath(os.path.join(base_dir, "Img_Dataset", filename)),
        # Try relative to current working directory
        os.path.join("Img_Dataset", filename),
        os.path.join("FRS", "Img_Dataset", filename),
        # Only try original path if it's an absolute path and doesn't contain "Img_Dataset" in a way that would duplicate
        normalized_path if (os.path.isabs(normalized_path) and "Img_Dataset" not in os.path.dirname(normalized_path)) else None,
    ]
    
    # Remove None values
    possible_paths = [p for p in possible_paths if p is not None]
    
    # Try each path
    for path in possible_paths:
        # Normalize the path to handle any remaining backslash issues
        path = os.path.normpath(path)
        if os.path.exists(path) and os.path.isfile(path):
            try:
                # Try to open the image to verify it's valid
                # Don't use verify() as it closes the file and can cause issues
                test_img = Image.open(path)
                test_img.load()  # Load the image to check if it's valid
                test_img.close()
                return path
            except Exception:
                # If opening fails, continue to next path
                continue
    
    # If none found, return the most likely path (will show error but at least tries)
    return os.path.join(base_dir, "Img_Dataset", filename)

# Create two columns: main content and history
main_col, history_col = st.columns([3, 1])

with main_col:
    # Let the user upload an image from their device via the browser
    uploaded_file = st.file_uploader("Upload a fashion product image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        saved_path = save_file(uploaded_file)
        if saved_path:
            try:
                # display image
                show_images = Image.open(uploaded_file)
                size = (400, 400)
                resized_im = show_images.resize(size)
                st.markdown("### Uploaded Image")
                st.image(resized_im, use_container_width=True)
                # extract features of uploaded image
                features = extract_img_features(saved_path, model)
                #st.text(features)
                img_indicess = recommendd(features, features_list)
                
                # Store in history
                history_entry = {
                    'uploaded_image': uploaded_file.name,
                    'uploaded_image_path': saved_path,
                    'recommendations': [img_files_list[img_indicess[0][i]] for i in range(5)],
                    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.history.insert(0, history_entry)
                # Keep only last 10 entries
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[:10]
                
                st.markdown("### Recommended Items")
                col1,col2,col3,col4,col5 = st.columns(5)

                # Display recommended images
                for idx, col in enumerate([col1, col2, col3, col4, col5]):
                    with col:
                        roman_num = ["I", "II", "III", "IV", "V"][idx]
                        st.markdown(f"<h3 style='color: #ffffff; text-align: center; margin-bottom: 10px;'>{roman_num}</h3>", unsafe_allow_html=True)
                        try:
                            img_path_from_list = img_files_list[img_indicess[0][idx]]
                            
                            # Resolve the actual image path
                            img_path = get_image_path(img_path_from_list)
                            
                            # Try to open and display the image
                            try:
                                rec_img = Image.open(img_path)
                                st.image(rec_img, use_container_width=True, clamp=True)
                            except Exception as img_error:
                                # If get_image_path returned a path that doesn't work, 
                                # try one more time with just the filename (extracted properly)
                                path_str = str(img_path_from_list).replace('\\', '/')
                                path_parts = path_str.split('/')
                                
                                # Extract filename - if path contains "Img_Dataset", extract just the filename part
                                if len(path_parts) > 1 and path_parts[0].lower() == "img_dataset":
                                    filename = path_parts[-1]
                                else:
                                    filename = os.path.basename(os.path.normpath(str(img_path_from_list)))
                                
                                final_path = os.path.join(BASE_DIR, "Img_Dataset", filename)
                                
                                if os.path.exists(final_path):
                                    try:
                                        rec_img = Image.open(final_path)
                                        st.image(rec_img, use_container_width=True, clamp=True)
                                    except Exception:
                                        st.error(f"Image found but cannot be opened: {filename}")
                                else:
                                    st.error(f"Image not found: {filename}")
                                    st.caption(f"Expected at: {final_path}")
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please try uploading a different image or check the console for more details.")
        else:
            st.error("❌ Failed to save uploaded file. Please try again.")
            st.info("Make sure you have proper permissions and the file is a valid image format (JPG, JPEG, PNG).")

with history_col:
    st.markdown("""
    <style>
    .history-container {
        background: rgba(26, 26, 46, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="history-container">', unsafe_allow_html=True)
    st.markdown("### 📜 History")
    
    if len(st.session_state.history) == 0:
        st.markdown("<p style='color: #ffffff; opacity: 0.7;'>No history yet</p>", unsafe_allow_html=True)
    else:
        for idx, entry in enumerate(st.session_state.history):
            with st.expander(f"📸 {entry['uploaded_image'][:20]}...", expanded=False):
                st.markdown(f"<p style='color: #ffffff; font-size: 0.8rem;'>{entry['timestamp']}</p>", unsafe_allow_html=True)
                try:
                    if os.path.exists(entry['uploaded_image_path']):
                        hist_img = Image.open(entry['uploaded_image_path'])
                        st.image(hist_img, use_container_width=True)
                        st.markdown("**Recommendations:**")
                        for i, rec_img in enumerate(entry['recommendations'][:3], 1):
                            st.image(rec_img, use_container_width=True, caption=f"Rec {i}")
                except Exception as e:
                    st.markdown(f"<p style='color: #ff6b6b;'>Image not found</p>", unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
