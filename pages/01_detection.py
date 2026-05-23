import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from utils.exif import get_image_metadata
from data.db import save_detection
from models.loader import load_detection_model, load_segmentation_model
from detection.predict import run_detection, run_segmentation, COCO_CLASSES

st.title("Detection")

uploaded_files = st.file_uploader(
    "Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        metadata = get_image_metadata(image)
        if metadata:
            st.subheader("Image Metadata")
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("No EXIF metadata found.")

        # Stage 1 — detection always runs
        with st.spinner("Running detection..."):
            det_img, detections = run_detection(load_detection_model(), image, COCO_CLASSES)

        if not detections:
            st.info("No objects detected in this image.")
            st.image(image, caption="No detections", use_container_width=True)
            save_detection(
                filename=uploaded_file.name,
                detections=[],
                metadata=metadata,
                confidence=0.0,
                mode="Detection",
            )
            continue

        # Stage 2 — segmentation only runs if detection found something (lazy cascade)
        with st.spinner("Running segmentation..."):
            seg_img, seg_detections = run_segmentation(load_segmentation_model(), image, COCO_CLASSES)

        st.subheader(f"Detections ({len(detections)})")
        for i, d in enumerate(detections):
            cls_name = COCO_CLASSES.get(d["class_id"], f"class_{d['class_id']}")
            x, y, w, h = d["bbox"]
            st.write(f"  **{i+1}.** {cls_name} — Conf: {d['confidence']:.2f} — Box: [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}]")

        st.image(det_img, caption="Detection (bounding boxes)", use_container_width=True)
        st.image(seg_img, caption="Segmentation (pixel masks)", use_container_width=True)

        save_detection(
            filename=uploaded_file.name,
            detections=detections,
            metadata=metadata,
            confidence=max((d["confidence"] for d in detections), default=0.0),
            mode="Detection + Segmentation",
        )

    st.success(f"Processed {len(uploaded_files)} image(s).")
