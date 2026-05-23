import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from utils.exif import get_image_metadata
from data.db import save_detection

st.title("🔍 Detection")

uploaded_files = st.file_uploader(
    "Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

        metadata = get_image_metadata(image)
        if metadata:
            st.subheader("Image Metadata")
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("No metadata found.")

        save_detection(
            filename=uploaded_file.name,
            detections=[],
            metadata=metadata,
            confidence=0.0,
            mode="Detection",
        )

    st.success(f"{len(uploaded_files)} image(s) uploaded and displayed successfully!")