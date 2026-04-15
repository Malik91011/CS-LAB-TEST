import streamlit as st
import cv2
import numpy as np
import urllib.request
import os
from PIL import Image

st.set_page_config(page_title="AI Vision", layout="wide")
st.title("🤖 CV Object Detector (MobileNet-SSD)")

# --- MODEL SETUP ---
# Path to save the model files
PROTOTXT = "deploy.prototxt"
MODEL_FILE = "mobilenet_iter_73000.caffemodel"

# Download files if they don't exist
def download_model():
    base_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
    if not os.path.exists(PROTOTXT):
        urllib.request.urlretrieve(base_url + "deploy.prototxt", PROTOTXT)
    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve("https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel", MODEL_FILE)

download_model()

@st.cache_resource
def load_net():
    return cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_FILE)

net = load_net()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# --- APP INTERFACE ---
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
conf_limit = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file)
    frame = np.array(image)
    (h, w) = frame.shape[:2]

    # Pre-process image for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_limit:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            
            # Box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw on image
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(frame, caption="Processed Image", use_container_width=True)
else:
    st.info("Upload an image to start detection.")
