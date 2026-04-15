import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Computer Vision Fix", layout="wide")

st.title("📷 Stable Computer Vision App")
st.write("This app uses `opencv-python-headless` for cloud compatibility.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process: Edge Detection (Proves cv2 is working)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Show Results
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.header("AI Edge Detection")
        st.image(edges, caption="OpenCV Success!", use_container_width=True)
else:
    st.info("Upload an image to verify the fix.")
