import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize Mediapipe Object Detection
mp_drawing = mp.solutions.drawing_utils
mp_object_detection = mp.solutions.object_detection

st.set_page_config(page_title="AI Vision", layout="wide")
st.title("🎯 AI Object Detector")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Run Mediapipe Detection
    with mp_object_detection.ObjectDetection(min_detection_confidence=0.4) as object_detection:
        # Convert RGB to BGR for OpenCV processing if necessary, 
        # but Mediapipe works well with RGB.
        results = object_detection.process(image_np)

        # Draw detections
        annotated_image = image_np.copy()
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)
                
                # Get label info
                label = detection.label_id
                score = detection.score[0]
                st.write(f"✅ Detected object with {score:.2f} confidence.")

        # Display results
        st.image(annotated_image, caption="AI Analysis Complete", use_container_width=True)

else:
    st.info("Please upload an image in the sidebar.")
