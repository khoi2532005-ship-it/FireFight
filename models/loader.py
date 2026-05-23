import streamlit as st
from ultralytics import YOLO

DETECTION_MODEL    = "models/weights/yolov8s.pt"
SEGMENTATION_MODEL = "models/weights/yolov8s-seg.pt"


@st.cache_resource
def load_detection_model():
    return YOLO(DETECTION_MODEL)


@st.cache_resource
def load_segmentation_model():
    return YOLO(SEGMENTATION_MODEL)
