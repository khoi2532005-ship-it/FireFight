import numpy as np
from PIL import Image

def run_detection(model, image: Image.Image):
    results = model(np.array(image))[0]
    result_img = Image.fromarray(results.plot())
    detections = [
        {
            "class": model.names[int(b.cls)],
            "confidence": float(b.conf),
            "bbox": b.xyxy[0].tolist(),
        }
        for b in results.boxes
    ] if results.boxes else []
    return result_img, detections

def run_segmentation(model, image: Image.Image):
    results = model(np.array(image))[0]
    result_img = Image.fromarray(results.plot())
    detections = [
        {
            "class": model.names[int(b.cls)],
            "confidence": float(b.conf),
        }
        for b in results.boxes
    ] if results.boxes else []
    return result_img, detections