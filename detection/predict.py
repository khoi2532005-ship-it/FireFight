import cv2
import numpy as np
from PIL import Image as PilImage

MODEL_INPUT_SIZE = 640

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
IOU_THRESH  = 0.45
NM = 32


def letterbox(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right  = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, left), r


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pad, r = letterbox(img)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    return img[np.newaxis], pad, r


def _nms(boxes, scores, class_ids, mask_coeffs):
    """Run per-class NMS and return aligned (detections, coeffs)."""
    if not boxes:
        return [], []

    # cv2.dnn.NMSBoxes needs list-of-list boxes and list-of-float scores
    boxes_cv  = [[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes]
    scores_cv = [float(s) for s in scores]

    kept = cv2.dnn.NMSBoxes(boxes_cv, scores_cv, CONF_THRESH, IOU_THRESH)
    if len(kept) == 0:
        return [], []

    kept = kept.flatten() if isinstance(kept, np.ndarray) else list(kept)

    detections, coeffs = [], []
    for i in kept:
        detections.append({
            "class_id":   int(class_ids[i]),
            "confidence": float(scores[i]),
            "bbox":       boxes[i],
        })
        coeffs.append(mask_coeffs[i])
    return detections, coeffs


def draw_detections(img, detections, class_names):
    for det in detections:
        x, y, w, h = det["bbox"]
        name  = class_names.get(det["class_id"], f"class_{det['class_id']}")
        color = (255, 69, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{name}: {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y - 10 if y - 10 > th else y + 10
        cv2.rectangle(img, (x, ty - th), (x + tw, ty + th), color, cv2.FILLED)
        cv2.putText(img, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_detection(sess, image, class_names=None):
    if class_names is None:
        class_names = COCO_CLASSES

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_input, pad, r = preprocess(img_bgr)
    ih, iw = img_bgr.shape[:2]
    gain = min(640 / ih, 640 / iw)

    input_name = sess.get_inputs()[0].name
    raw = sess.run(None, {input_name: img_input})
    preds = np.transpose(np.squeeze(raw[0]))   # (8400, 84)

    boxes, scores, class_ids = [], [], []
    for row in preds:
        probs    = row[4:]
        score    = float(np.max(probs))
        if score < CONF_THRESH:
            continue
        cls      = int(np.argmax(probs))
        cx, cy, w, h = row[:4]
        left = int((cx - w / 2 - pad[1]) / gain)
        top  = int((cy - h / 2 - pad[0]) / gain)
        boxes.append([left, top, int(w / gain), int(h / gain)])
        scores.append(score)
        class_ids.append(cls)

    detections, _ = _nms(boxes, scores, class_ids, [None] * len(boxes))

    result = img_bgr.copy()
    draw_detections(result, detections, class_names)
    result_pil = PilImage.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_pil, [{"class_id": d["class_id"], "confidence": d["confidence"],
                         "bbox": d["bbox"]} for d in detections]


def run_segmentation(sess, image, class_names=None):
    if class_names is None:
        class_names = COCO_CLASSES

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_input, pad, r = preprocess(img_bgr)
    ih, iw = img_bgr.shape[:2]
    gain = min(640 / ih, 640 / iw)

    input_name  = sess.get_inputs()[0].name
    out_names   = [o.name for o in sess.get_outputs()]
    raw         = sess.run(out_names, {input_name: img_input})

    preds  = raw[0]   # (1, 4+nm+80, 8400)
    protos = raw[1]   # (1, nm, 160, 160)

    nm      = protos.shape[1]        # 32
    proto_h = protos.shape[2]        # 160
    proto_w = protos.shape[3]        # 160

    # Letterbox padding mapped to proto space
    scale_p       = proto_h / 640.0
    proto_pad_top = int(pad[0] * scale_p)
    proto_pad_lft = int(pad[1] * scale_p)
    act_h = proto_h - 2 * proto_pad_top
    act_w = proto_w - 2 * proto_pad_lft

    boxes, scores, class_ids, coeffs = [], [], [], []
    for i in range(preds.shape[2]):
        row   = preds[0, :, i]
        probs = row[4 + nm:]
        score = float(np.max(probs))
        if score < CONF_THRESH:
            continue
        cls      = int(np.argmax(probs))
        cx, cy, w, h = row[:4]
        left = int((cx - w / 2 - pad[1]) / gain)
        top  = int((cy - h / 2 - pad[0]) / gain)
        boxes.append([left, top, int(w / gain), int(h / gain)])
        scores.append(score)
        class_ids.append(cls)
        coeffs.append(row[4:4 + nm].astype(np.float32))

    detections, kept_coeffs = _nms(boxes, scores, class_ids, coeffs)

    result = img_bgr.copy()

    if detections:
        protos_np = protos[0].astype(np.float32)  # (nm, 160, 160)

        # Crop to active (non-padded) proto region
        proto_crop = protos_np[
            :,
            proto_pad_top: proto_pad_top + act_h,
            proto_pad_lft: proto_pad_lft + act_w,
        ]  # (nm, act_h, act_w)

        coeff_arr = np.stack(kept_coeffs)          # (N, nm)
        ph, pw    = proto_crop.shape[1], proto_crop.shape[2]

        # Matrix multiply -> sigmoid -> (N, ph, pw)
        raw_masks = coeff_arr @ proto_crop.reshape(nm, -1)   # (N, ph*pw)
        raw_masks = raw_masks.reshape(-1, ph, pw)
        masks_sig = 1.0 / (1.0 + np.exp(-raw_masks))        # sigmoid

        # Resize each mask to original image dimensions
        masks_full = np.stack([
            cv2.resize(masks_sig[k], (iw, ih), interpolation=cv2.INTER_LINEAR)
            for k in range(masks_sig.shape[0])
        ])  # (N, ih, iw)

        green = np.array([0, 200, 0], dtype=np.float32)

        for idx, det in enumerate(detections):
            x, y, bw, bh = det["bbox"]
            x1, y1 = max(0, x),      max(0, y)
            x2, y2 = min(iw, x + bw), min(ih, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop mask to bbox, threshold at 0.5
            mask_roi = (masks_full[idx, y1:y2, x1:x2] > 0.5)[..., np.newaxis]
            roi      = result[y1:y2, x1:x2].astype(np.float32)
            overlay  = roi * 0.55 + green * 0.45
            result[y1:y2, x1:x2] = np.where(mask_roi, overlay, roi).astype(np.uint8)

    result_pil = PilImage.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_pil, [{"class_id": d["class_id"], "confidence": d["confidence"],
                         "bbox": d["bbox"]} for d in detections]
