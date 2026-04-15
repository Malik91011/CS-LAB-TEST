import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Vision Explorer", layout="wide")

st.title("📷 AI Computer Vision App")
st.write("Upload an image to detect objects in real-time using a pretrained YOLOv8 model.")

# Load the model (this will download the 'yolov8n.pt' file automatically on first run)
# 'n' stands for nano, which is the fastest and most lightweight version for web apps.
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

# Sidebar for settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image PIL can read
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.header("AI Detection")
        
        # Perform detection
        # We convert PIL image to numpy array for YOLO
        results = model.predict(source=image, conf=confidence_threshold)
        
        # Plot the results on the image
        # res[0].plot() returns a BGR numpy array
        res_plotted = results[0].plot()[:, :, ::-1] 
        
        st.image(res_plotted, caption="Detected Objects", use_container_width=True)

    # Display raw data summary
    st.subheader("Detection Results")
    detections = results[0].boxes.data.tolist()
    if detections:
        for det in detections:
            # x1, y1, x2, y2, confidence, class_id
            class_id = int(det[5])
            label = model.names[class_id]
            score = det[4]
            st.write(f"✅ Found **{label}** with {score:.2f} confidence")
    else:
        st.write("No objects detected. Try lowering the confidence threshold.")

else:
    st.info("Please upload an image file in the sidebar to begin.")
