import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(page_title="AI Emotion Detector")

st.title("🎭 AI Emotion Recognition")
st.write("Detecting feelings (Happy, Sad, Angry, etc.) using DeepFace.")

uploaded_file = st.file_uploader("Upload a clear photo of a face", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load and Save Image temporarily
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    
    # Save to a temp file because DeepFace needs a file path
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    try:
        with st.spinner('AI is analyzing the face...'):
            # 2. Run DeepFace Analysis
            # enforce_detection=False prevents crashing if a face isn't perfectly clear
            results = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)

        # 3. Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Analysis Results")
            # DeepFace returns a list (in case there are multiple faces)
            main_face = results[0]
            dominant_emotion = main_face['dominant_emotion']
            confidence = main_face['emotion'][dominant_emotion]

            st.metric(label="Detected Emotion", value=dominant_emotion.upper())
            st.progress(confidence / 100)
            st.write(f"Confidence Score: **{confidence:.2f}%**")

            with st.expander("See all emotion levels"):
                for emotion, score in main_face['emotion'].items():
                    st.write(f"{emotion.title()}: {score:.1f}%")

    except Exception as e:
        st.error(f"AI Analysis failed: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("Please upload an image to begin.")
