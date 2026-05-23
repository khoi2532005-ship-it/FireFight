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

# Cache results in session_state keyed by filename to avoid reprocessing on rerun
if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}

# Clear cache when file list changes
current_names = {f.name for f in uploaded_files} if uploaded_files else set()
cached_names  = set(st.session_state.results_cache.keys())
if current_names != cached_names:
    # Remove stale entries no longer in uploader
    for name in cached_names - current_names:
        del st.session_state.results_cache[name]

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        metadata = get_image_metadata(image)
        if metadata:
            st.subheader("Image Metadata")
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("No EXIF metadata found.")

        # Only run inference if not already cached for this file
        if uploaded_file.name not in st.session_state.results_cache:
            # Stage 1 — detection
            with st.spinner(f"Running detection on {uploaded_file.name}..."):
                det_img, detections = run_detection(load_detection_model(), image, COCO_CLASSES)

            if not detections:
                st.session_state.results_cache[uploaded_file.name] = {
                    "detections": [],
                    "det_img": None,
                    "seg_img": None,
                }
                save_detection(
                    filename=uploaded_file.name,
                    detections=[],
                    metadata=metadata,
                    confidence=0.0,
                    mode="Detection",
                )
            else:
                # Stage 2 — segmentation (lazy, only if detections found)
                with st.spinner(f"Running segmentation on {uploaded_file.name}..."):
                    seg_img, seg_detections = run_segmentation(load_segmentation_model(), image, COCO_CLASSES)

                st.session_state.results_cache[uploaded_file.name] = {
                    "detections": detections,
                    "det_img":    det_img,
                    "seg_img":    seg_img,
                }
                save_detection(
                    filename=uploaded_file.name,
                    detections=detections,
                    metadata=metadata,
                    confidence=max((d["confidence"] for d in detections), default=0.0),
                    mode="Detection + Segmentation",
                )

        # Display cached results
        cached = st.session_state.results_cache.get(uploaded_file.name, {})
        detections = cached.get("detections", [])

        if not detections:
            st.info("No objects detected in this image.")
            st.image(image, caption="No detections", use_container_width=True)
        else:
            st.subheader(f"Detections ({len(detections)})")
            for i, d in enumerate(detections):
                cls_name = COCO_CLASSES.get(d["class_id"], f"class_{d['class_id']}")
                x, y, w, h = d["bbox"]
                st.write(f"  **{i+1}.** {cls_name} — Conf: {d['confidence']:.2f} — Box: [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}]")

            st.image(cached["det_img"], caption="Detection (bounding boxes)", use_container_width=True)
            st.image(cached["seg_img"], caption="Segmentation (pixel masks)", use_container_width=True)

    st.success(f"Processed {len(uploaded_files)} image(s).")
