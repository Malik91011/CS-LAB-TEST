import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# Page Config
st.set_page_config(page_title="AI Vision Fix", layout="wide")

st.title("🛡️ Error-Free Vision App")
st.write("Current Environment: **Python 3.14 (Stable Mode)**")

# Sidebar
st.sidebar.header("Controls")
filter_type = st.sidebar.selectbox(
    "Select AI Filter:",
    ["Original", "Grayscale", "Find Edges", "Contour", "Blur", "Sharpen"]
)

uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # 1. Load Image
        img = Image.open(uploaded_file)
        
        # 2. FIX: Convert to RGB (Removes transparency/alpha channel)
        # This prevents the 'ValueError' in Pillow filters
        img = img.convert("RGB")
        
        # 3. Apply Filters Defensively
        processed_img = img.copy()
        
        if filter_type == "Grayscale":
            processed_img = ImageOps.grayscale(img)
        elif filter_type == "Find Edges":
            processed_img = img.filter(ImageFilter.FIND_EDGES)
        elif filter_type == "Contour":
            processed_img = img.filter(ImageFilter.CONTOUR)
        elif filter_type == "Blur":
            processed_img = img.filter(ImageFilter.BLUR)
        elif filter_type == "Sharpen":
            processed_img = img.filter(ImageFilter.SHARPEN)

        # 4. Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(img, use_container_width=True)
        with col2:
            st.subheader(f"Output: {filter_type}")
            st.image(processed_img, use_container_width=True)

        # Analysis Table
        st.divider()
        st.subheader("Image Metadata")
        st.write(f"**Mode:** {img.mode} | **Size:** {img.size[0]}x{img.size[1]}px")

    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
        st.info("Try uploading a different image file.")

else:
    st.info("Please upload an image to begin.")
