import streamlit as st
from fer import FER
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Emotion Detector")
st.title("🎭 AI Emotion Recognition")

# Cache the model so it doesn't reload on every click
@st.cache_resource
def load_detector():
    return FER(mtcnn=True)

detector = load_detector()

uploaded_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    with st.spinner('Analyzing emotions...'):
        results = detector.detect_emotions(image_np)
    
    if results:
        top_emotion = max(results[0]["emotions"], key=results[0]["emotions"].get)
        confidence = results[0]["emotions"][top_emotion]
        
        st.image(image, use_container_width=True)
        st.success(f"The AI thinks this person feels **{top_emotion.upper()}** ({confidence*100:.1f}% confidence)")
    else:
        st.warning("Could not find a face in this image. Try a clearer shot!")
