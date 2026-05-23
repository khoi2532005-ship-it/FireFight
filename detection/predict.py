import cv2
import numpy as np
from PIL import Image as PilImage

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

CONF_THRESH = 0.45


def run_detection(model, image, class_names=None):
    if class_names is None:
        class_names = COCO_CLASSES

    results   = model(image, conf=CONF_THRESH, verbose=False)[0]
    img_bgr   = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_img = img_bgr.copy()

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            name  = class_names.get(cls, f"class_{cls}")
            color = (255, 69, 0)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = y1 - 10 if y1 - 10 > th else y1 + 10
            cv2.rectangle(result_img, (x1, ty - th), (x1 + tw, ty + th), color, cv2.FILLED)
            cv2.putText(result_img, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            detections.append({"class_id": cls, "confidence": conf,
                                "bbox": [x1, y1, x2 - x1, y2 - y1]})

    result_pil = PilImage.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    return result_pil, detections


def run_segmentation(model, image, class_names=None):
    if class_names is None:
        class_names = COCO_CLASSES

    results    = model(image, conf=CONF_THRESH, verbose=False)[0]
    img_bgr    = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_img = img_bgr.copy()
    ih, iw     = img_bgr.shape[:2]

    detections = []
    if results.masks is not None and results.boxes is not None:
        green = np.array([0, 200, 0], dtype=np.float32)

        for box, mask in zip(results.boxes, results.masks):
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Resize mask to original image size
            mask_np      = mask.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_np, (iw, ih), interpolation=cv2.INTER_LINEAR)
            mask_bin     = (mask_resized > 0.5)[..., np.newaxis]

            roi        = result_img.astype(np.float32)
            result_img = np.where(mask_bin, roi * 0.55 + green * 0.45, roi).astype(np.uint8)

            detections.append({"class_id": cls, "confidence": conf,
                                "bbox": [x1, y1, x2 - x1, y2 - y1]})

    result_pil = PilImage.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    return result_pil, detections
