import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from utils.exif import get_image_metadata
from data.db import save_detection
from models.loader import load_detection_model, load_segmentation_model
from detection.predict import run_detection, run_segmentation, COCO_CLASSES

st.title("Detection")

if "det_models_loaded" not in st.session_state:
    st.session_state.det_models_loaded = False
if "seg_models_loaded" not in st.session_state:
    st.session_state.seg_models_loaded = False

if not st.session_state.det_models_loaded:
    with st.spinner("Loading detection model..."):
        st.session_state.det_model = load_detection_model()
        st.session_state.det_models_loaded = True

if not st.session_state.seg_models_loaded:
    with st.spinner("Loading segmentation model..."):
        st.session_state.seg_model = load_segmentation_model()
        st.session_state.seg_models_loaded = True

uploaded_files = st.file_uploader(
    "Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        metadata = get_image_metadata(image)
        if metadata:
            st.subheader("Image Metadata")
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("No EXIF metadata found.")

        with st.spinner("Running detection..."):
            det_img, detections = run_detection(st.session_state.det_model, image, COCO_CLASSES)

        with st.spinner("Running segmentation..."):
            seg_img, seg_detections = run_segmentation(st.session_state.seg_model, image, COCO_CLASSES)

        if detections:
            st.subheader(f"Detections ({len(detections)})")
            for i, d in enumerate(detections):
                cls_name = COCO_CLASSES.get(d["class_id"], f"class_{d['class_id']}")
                x, y, w, h = d["bbox"]
                st.write(f"  **{i+1}.** {cls_name} — Conf: {d['confidence']:.2f} — Box: [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}]")

            st.image(det_img, caption="Detection (bounding boxes)", use_container_width=True)
            st.image(seg_img, caption="Segmentation (pixel masks)", use_container_width=True)
        else:
            st.info("No objects detected in this image.")
            st.image(image, caption="No detections", use_container_width=True)

        save_detection(
            filename=uploaded_file.name,
            detections=detections,
            metadata=metadata,
            confidence=max((d["confidence"] for d in detections), default=0.0),
            mode="Detection + Segmentation",
        )

    st.success(f"Processed {len(uploaded_files)} image(s).")
