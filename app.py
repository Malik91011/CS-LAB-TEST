import streamlit as st
from fer import FER
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="AI Emotion Detector", layout="wide")

st.title("🎭 AI Facial Emotion Recognition")
st.write("Upload a photo of a face, and the AI will tell you how they feel.")

# Initialize the Emotion Detector
# mtcnn=True makes it more accurate but slightly slower
@st.cache_resource
def load_detector():
    return FER(mtcnn=True)

detector = load_detector()

uploaded_file = st.file_uploader("Choose a clear photo of a face...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load Image
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))

    # 2. Run Detection
    with st.spinner('AI is analyzing facial expressions...'):
        # detector.detect_emotions returns a list of faces and their scores
        results = detector.detect_emotions(image_np)

    if results:
        # Create columns for the image and the results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="Analyzed Image", use_container_width=True)

        with col2:
            st.subheader("Emotion Analysis")
            for face in results:
                # Get the box where the face is
                box = face["box"]
                # Get the emotions dictionary
                emotions = face["emotions"]
                
                # Find the emotion with the highest score
                top_emotion = max(emotions, key=emotions.get)
                confidence = emotions[top_emotion] * 100

                # Display the result with a nice color highlight
                st.metric(label="Primary Emotion", value=top_emotion.title())
                st.write(f"Confidence: **{confidence:.1f}%**")
                
                # Show all detected scores for this face
                with st.expander("Show detailed scores"):
                    for emo, score in emotions.items():
                        st.write(f"{emo.title()}: {score:.2f}")
    else:
        st.warning("No faces detected. Please try a clearer photo or a front-facing shot.")

else:
    st.info("Please upload an image to start.")
