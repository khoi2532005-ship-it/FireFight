import cv2
import numpy as np

MODEL_INPUT_SIZE = 640

# COCO 80 classes
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

# YOLOv8-seg ONNX output layout: [1, 116, 8400]
#   4:4+nm  = mask coefficients (32)
#   4+nm:   = class probabilities (80)
NM = 32


def letterbox(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, left), r


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pad, r = letterbox(img)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    img = img[np.newaxis, ...]
    return img, pad, r


def postprocess_nms(boxes, scores, class_ids, conf_thres=0.25, iou_thres=0.7):
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) > 0:
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        detections = []
        for i in indices:
            detections.append({
                "class_id": class_ids[int(i)],
                "confidence": float(scores[int(i)]),
                "bbox": boxes[int(i)],
            })
        return detections
    return []


def draw_detections(img, detections, class_names):
    for det in detections:
        x, y, w, h = det["bbox"]
        score = det["confidence"]
        cls = det["class_id"]
        name = class_names.get(cls, f"class_{cls}")
        color = (255, 69, 0)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        label = f"{name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y - 10 if y - 10 > th else y + 10
        cv2.rectangle(img, (int(x), int(ty - th)), (int(x + tw), int(ty + th)), color, cv2.FILLED)
        cv2.putText(img, label, (int(x), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_detection(sess, image, class_names=None):
    if class_names is None:
        class_names = COCO_CLASSES

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_input, pad, r = preprocess(img_bgr)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: img_input})
    img_shape = img_bgr.shape[:2]

    outputs = np.transpose(np.squeeze(output))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    gain = min(640 / img_shape[0], 640 / img_shape[1])
    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= 0.25:
            class_id = int(np.argmax(classes_scores))
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    detections = postprocess_nms(boxes, scores, class_ids)

    result_img = img_bgr.copy()
    result_img = draw_detections(result_img, detections, class_names)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    detections_list = [
        {"class_id": d["class_id"], "confidence": d["confidence"], "bbox": d["bbox"]}
        for d in detections
    ]

    return image.copy(), detections_list


def run_segmentation(sess, image, class_names=None):
    if class_names is None:
        class_names = COCO_CLASSES

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_input, pad, r = preprocess(img_bgr)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, {input_name: img_input})

    preds = outputs[0]  # [1, 116, 8400]
    protos = outputs[1]  # [1, 32, 160, 160]

    img_shape = img_bgr.shape[:2]
    nm = protos.shape[0]  # 32

    # preds layout: [1, 4+bbox, nm+4, nc+nm+4] -> [1, 4, 32, 80] per detection
    # For each detection i:
    #   preds[0, :4, i]       = bbox (x, y, w, h)
    #   preds[0, 4:4+nm, i]   = mask coefficients
    #   preds[0, 4+nm:, i]    = class probabilities (80)

    boxes = []
    scores = []
    class_ids = []
    mask_coeffs = []

    gain = min(640 / img_shape[0], 640 / img_shape[1])
    for i in range(preds.shape[2]):
        row = preds[0, :, i]
        bbox = row[:4]
        coeffs = row[4:4+nm]
        class_probs = row[4+nm:]

        max_score = float(np.max(class_probs))
        if max_score >= 0.25:
            cls = int(np.argmax(class_probs))
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            left = int((x - w / 2 - pad[1]) / gain)
            top = int((y - h / 2 - pad[0]) / gain)
            width = int(w / gain)
            height = int(h / gain)
            class_ids.append(cls)
            scores.append(max_score)
            boxes.append([left, top, width, height])
            mask_coeffs.append(coeffs)

    detections = postprocess_nms(boxes, scores, class_ids)

    result_img = img_bgr.copy()
    if detections and mask_coeffs:
        mask_coeffs = np.array(mask_coeffs)
        protos_float = protos.astype(np.float32)[0]  # [32, 160, 160]

        # (N, 32) @ (32, 160*160) = (N, 160, 160)
        masks_flat = (mask_coeffs @ protos_float.reshape(nm, -1)).reshape(-1, protos.shape[2], protos.shape[3])
        masks = 1.0 / (1.0 + np.exp(-masks_flat))  # sigmoid

        # Scale masks to original image size
        mh, mw = masks.shape[1], masks.shape[2]
        target_h, target_w = img_shape[0], img_shape[1]
        masks = cv2.resize(masks, (target_w, target_h))

        # Apply each mask to its corresponding detection
        for idx, det in enumerate(detections):
            x, y, bw, bh = det["bbox"]
            cls = det["class_id"]
            score = det["confidence"]
            mask = masks[idx, y:y+bh, x:x+bw]
            mask = (mask > 0.5).astype(np.uint8)

            roi = result_img[y:y+bh, x:x+bw].astype(np.float32)
            green = np.array([0, 200, 0], dtype=np.float32)
            overlay = roi * 0.55 + green * 0.45
            overlay = np.where(mask[..., np.newaxis] > 0, overlay, roi)
            result_img[y:y+bh, x:x+bw] = overlay.astype(np.uint8)

            # Draw class name label on the mask
            name = class_names.get(cls, f"class_{cls}")
            label = f"{name}: {score:.2f}"
            color = (0, 200, 0)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y - 12 if y - 12 > th else y + 12
            cv2.rectangle(result_img, (int(x), int(ty - th)), (int(x + tw), int(ty + th)), color, cv2.FILLED)
            cv2.putText(result_img, label, (int(x), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    result_img = draw_detections(result_img, detections, class_names)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    detections_list = [
        {"class_id": d["class_id"], "confidence": d["confidence"], "bbox": d["bbox"]}
        for d in detections
    ]

    return image.copy(), detections_list
