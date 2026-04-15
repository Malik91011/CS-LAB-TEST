import streamlit as st
from rembg import remove
from PIL import Image
import io

# Page setup
st.set_page_config(page_title="AI Background Remover", layout="centered")

st.title("✂️ AI Background Remover")
st.write("Upload an image to remove the background instantly using the U2-Net model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load the original image
    input_image = Image.open(uploaded_file)
    
    # Create columns for side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original")
        st.image(input_image, use_container_width=True)

    with col2:
        st.header("Background Removed")
        with st.spinner('AI is processing...'):
            # 2. Run the Background Removal Model
            # The 'remove' function handles the math and transparency
            output_image = remove(input_image)
            st.image(output_image, use_container_width=True)

    # 3. Download Button
    # We must convert the result back to bytes for the download button
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Transparent Image",
        data=byte_im,
        file_name="background_removed.png",
        mime="image/png"
    )

else:
    st.info("Please upload an image to see the AI in action.")
