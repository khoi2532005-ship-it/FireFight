import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from utils.exif import get_image_metadata
from data.db import save_detection, clear_db
from models.loader import load_detection_model, load_segmentation_model
from detection.predict import (
    run_detection, run_segmentation,
    DETECTION_CLASSES, SEGMENTATION_CLASSES,
)

st.sidebar.title("🔥 Fire Fight")

if st.sidebar.button("🗑️ Clear Database & Map", use_container_width=True):
    clear_db()
    if "results_cache" in st.session_state:
        st.session_state.results_cache.clear()
    st.rerun()

st.title("Detection")

# Model names (display purposes)
DETECTION_MODEL_NAME = "YOLO Fire/Smoke Detector (5-class)"
SEGMENTATION_MODEL_NAME = "YOLO Fire/Smoke Segmentation Model (2-class)"

uploaded_files = st.file_uploader(
    "Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}

current_names = {f.name for f in uploaded_files} if uploaded_files else set()
cached_names  = set(st.session_state.results_cache.keys())

if current_names != cached_names:
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
            metadata = {}
            st.write("No EXIF metadata found.")

        if uploaded_file.name not in st.session_state.results_cache:
            if "Latitude" not in metadata or "Longitude" not in metadata:
                st.warning("No GPS coordinates found in image. Please enter them manually for mapping.")
                col1, col2 = st.columns(2)
                # Defaults to Sydney
                metadata["Latitude"] = col1.number_input("Latitude", value=-33.8688, format="%.6f", key=f"lat_{uploaded_file.name}")
                metadata["Longitude"] = col2.number_input("Longitude", value=151.2093, format="%.6f", key=f"lon_{uploaded_file.name}")
                
                if not st.button(f"Analyze {uploaded_file.name}", key=f"btn_{uploaded_file.name}"):
                    continue

        if uploaded_file.name not in st.session_state.results_cache:

            # Stage 1 — detection
            with st.spinner(f"Running detection on {uploaded_file.name}..."):
                det_model = load_detection_model()
                det_img, detections = run_detection(det_model, image, DETECTION_CLASSES)

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
                    risk_level=0
                )

            else:
                with st.spinner(f"Running segmentation on {uploaded_file.name}..."):
                    seg_model = load_segmentation_model()
                    seg_img, seg_detections, coverage = run_segmentation(
                        seg_model, image, SEGMENTATION_CLASSES
                    )
                
                if coverage >= 80:
                    risk_level = 3
                elif coverage >= 50:
                    risk_level = 2
                elif coverage >= 20:
                    risk_level = 1
                else:
                    risk_level = 0

                st.session_state.results_cache[uploaded_file.name] = {
                    "detections": detections,
                    "det_img": det_img,
                    "seg_img": seg_img,
                    "coverage": coverage,
                    "risk_level": risk_level
                }

                save_detection(
                    filename=uploaded_file.name,
                    detections=detections,
                    metadata=metadata,
                    confidence=max((d["confidence"] for d in detections), default=0.0),
                    mode="Detection + Segmentation",
                    risk_level=risk_level
                )

        cached = st.session_state.results_cache.get(uploaded_file.name, {})
        detections = cached.get("detections", [])

        if not detections:
            st.info("No fire or smoke detected in this image.")
            st.image(image, caption="No detections", use_container_width=True)

        else:
            st.subheader(f"Detections ({len(detections)})")
            
            if "coverage" in cached:
                st.write(f"**Segmentation Coverage:** {cached['coverage']:.2f}%")
                st.write(f"**Assessed Risk Level:** {cached.get('risk_level', 0)}")

            for i, d in enumerate(detections):
                cls_name = DETECTION_CLASSES.get(d["class_id"], f"class_{d['class_id']}")

                if cls_name == "Fire":
                    cls_name = "Smoke"
                elif cls_name == "Smoke":
                    cls_name = "Fire"

                x, y, w, h = d["bbox"]

                st.write(
                    f"**{i+1}.** {cls_name} "
                    f"— Conf: {d['confidence']:.2f} "
                    f"— Box: [{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}]"
                )

            # SHOW MODEL OUTPUTS WITH MODEL NAMES
            st.image(
                cached["det_img"],
                caption=f"Detection Output — {DETECTION_MODEL_NAME}",
                use_container_width=True
            )

            st.image(
                cached["seg_img"],
                caption=f"Segmentation Output — {SEGMENTATION_MODEL_NAME}",
                use_container_width=True
            )

    st.success(f"Processed {len(uploaded_files)} image(s).")