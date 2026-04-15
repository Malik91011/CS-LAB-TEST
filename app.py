import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Page Config
st.set_page_config(page_title="AI Storyteller", layout="centered")

st.title("📖 AI Image Storyteller")
st.write("Upload an image, and the AI will describe what's happening.")

# Load the Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_captioning_model():
    # Using the 'base' model for faster performance on CPU
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_captioning_model()

uploaded_file = st.file_uploader("Select an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the image
    raw_image = Image.open(uploaded_file).convert('RGB')
    st.image(raw_image, caption="Uploaded Image", use_container_width=True)

    with st.spinner('Generating caption...'):
        # 2. Process the image for the model
        inputs = processor(raw_image, return_tensors="pt")

        # 3. Generate the text
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    # 4. Show the result
    st.subheader("AI Description:")
    st.success(caption.capitalize())
    
    # Extra: Make it a "Story"
    st.info(f"The AI sees: '{caption}' and thinks this would be a great start to a story!")

else:
    st.info("Please upload an image to see the AI Storyteller in action.")
