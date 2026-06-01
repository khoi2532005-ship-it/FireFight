import cv2
import numpy as np
from PIL import Image as PilImage

# Detection model (YOLOv8m from HuggingFace kittendev/YOLOv8m-smoke-detection) — 2 classes
DETECTION_CLASSES = {
    0: "smoke",
    1: "fire",
}

# Segmentation model (yolov8s-seg.pt) — 2 classes (unchanged)
SEGMENTATION_CLASSES = {
    0: "fire",
    1: "smoke",
}

# Per-class colours for detection (BGR)
DETECTION_COLORS = {
    1: (0,  69, 255),    # fire  – red-orange
    0: (180, 180, 180),  # smoke – light grey
}

# Per-class colours for segmentation overlay (RGB float)
SEGMENTATION_OVERLAY_COLORS = {
    0: np.array([255,  80,   0], dtype=np.float32),   # fire  – orange
    1: np.array([160, 160, 160], dtype=np.float32),   # smoke – grey
}

CONF_THRESH = 0.15


def run_detection(model, image, class_names=None):
    if class_names is None:
        class_names = DETECTION_CLASSES

    results    = model(image, conf=CONF_THRESH, verbose=False)[0]
    img_bgr    = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_img = img_bgr.copy()

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            name  = class_names.get(cls, f"class_{cls}")
            color = DETECTION_COLORS.get(cls, (255, 69, 0))

            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y1 - 10 if y1 - 10 > th else y1 + 10
            cv2.rectangle(result_img, (x1, ty - th), (x1 + tw, ty + th), color, cv2.FILLED)
            cv2.putText(result_img, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "class_id":   cls,
                "class_name": name,
                "confidence": conf,
                "bbox":       [x1, y1, x2 - x1, y2 - y1],
            })

    result_pil = PilImage.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    return result_pil, detections


def run_segmentation(model, image, class_names=None):
    if class_names is None:
        class_names = SEGMENTATION_CLASSES

    results    = model(image, conf=CONF_THRESH, verbose=False)[0]
    img_bgr    = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_img = img_bgr.copy()
    ih, iw     = img_bgr.shape[:2]

    DEFAULT_OVERLAY = np.array([0, 200, 0], dtype=np.float32)

    total_mask = np.zeros((ih, iw), dtype=bool)

    detections = []
    if results.masks is not None and results.boxes is not None:
        for box, mask in zip(results.boxes, results.masks):
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = class_names.get(cls, f"class_{cls}")

            mask_np      = mask.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_np, (iw, ih), interpolation=cv2.INTER_LINEAR)
            mask_bin     = (mask_resized > 0.5)
            
            total_mask   = np.logical_or(total_mask, mask_bin)
            mask_bin_expanded = mask_bin[..., np.newaxis]

            overlay    = SEGMENTATION_OVERLAY_COLORS.get(cls, DEFAULT_OVERLAY)
            roi        = result_img.astype(np.float32)
            result_img = np.where(mask_bin_expanded, roi * 0.55 + overlay * 0.45, roi).astype(np.uint8)

            color = (int(overlay[2]), int(overlay[1]), int(overlay[0]))  # RGB → BGR
            label = f"{name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y1 - 10 if y1 - 10 > th else y1 + 10
            cv2.rectangle(result_img, (x1, ty - th), (x1 + tw, ty + th), color, cv2.FILLED)
            cv2.putText(result_img, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "class_id":   cls,
                "class_name": name,
                "confidence": conf,
                "bbox":       [x1, y1, x2 - x1, y2 - y1],
            })

    coverage = 0.0
    if ih > 0 and iw > 0:
        coverage = (np.sum(total_mask) / (ih * iw)) * 100.0

    result_pil = PilImage.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    return result_pil, detections, coverage