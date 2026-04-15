import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np

st.set_page_config(page_title="Stable AI Vision", layout="wide")

st.title("🖼️ Stable Computer Vision App")
st.write("This version uses **Pillow** to avoid the OpenCV 'ImportError'.")

# Sidebar for Image Filters
st.sidebar.header("Filter Settings")
filter_type = st.sidebar.selectbox(
    "Choose a Vision Filter:",
    ["None", "Edge Enhancement", "Find Edges", "Grayscale", "Blur", "Contour"]
)

uploaded_file = st.file_uploader("Upload an image to process", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open image using Pillow
    img = Image.open(uploaded_file)
    
    # Process the image based on selection
    processed_img = img.copy()
    
    if filter_type == "Grayscale":
        processed_img = ImageOps.grayscale(img)
    elif filter_type == "Edge Enhancement":
        processed_img = img.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == "Find Edges":
        processed_img = img.filter(ImageFilter.FIND_EDGES)
    elif filter_type == "Blur":
        processed_img = img.filter(ImageFilter.BLUR)
    elif filter_type == "Contour":
        processed_img = img.filter(ImageFilter.CONTOUR)

    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original")
        st.image(img, use_container_width=True)
        
    with col2:
        st.header(f"Result: {filter_type}")
        st.image(processed_img, use_container_width=True)

    # Simple Computer Vision Data (Image Stats)
    st.subheader("Image Analysis")
    width, height = img.size
    img_array = np.array(img)
    st.write(f"📏 **Dimensions:** {width}x{height} pixels")
    st.write(f"🎨 **Color Mode:** {img.mode}")
    st.write(f"📊 **Average Brightness:** {np.mean(img_array):.2f}")

else:
    st.info("Upload an image to see the computer vision model in action.")
