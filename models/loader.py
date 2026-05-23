import streamlit as st
import onnxruntime as rt

DETECTION_MODEL = "models/weights/yolov8s.onnx"
SEGMENTATION_MODEL = "models/weights/yolov8s-seg.onnx"


@st.cache_resource
def load_detection_model():
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess = rt.InferenceSession(DETECTION_MODEL, sess_options, providers=["CPUExecutionProvider"])
    return sess


@st.cache_resource
def load_segmentation_model():
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess = rt.InferenceSession(SEGMENTATION_MODEL, sess_options, providers=["CPUExecutionProvider"])
    return sess
