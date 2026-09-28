import numpy as np
import cv2
from typing import List, Tuple
import time

from pipeline.ml_model import MLClassifier

def compute_iou(boxA, boxB):
    # box: (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / (area[i] + area[idxs[1:]] - w * h)
        
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > iou_threshold)[0] + 1)))
        
    return boxes[pick].tolist(), scores[pick].tolist()

def sliding_window_proposals(
    frame: np.ndarray,
    ml_model: MLClassifier,
    window_size: Tuple[int, int] = (64, 64),
    stride: int = 32,
    threshold: float = 0.5,
    nms_iou_thresh: float = 0.3
):
    """
    Lightweight sliding-window AI search over a frame.
    Returns: list of (bbox, ai_score)
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    h, w = frame.shape
    ww, wh = window_size
    
    boxes = []
    scores = []
    
    windows_evaluated = 0
    
    # For speed, we can process crops individually or batch them,
    # but the MLClassifier currently expects one crop per call in predict_crop.
    # To optimize this for sliding window, we could extract features in batch if needed,
    # but let's stick to predict_crop for simplicity first.
    
    for y in range(0, h - wh + 1, stride):
        for x in range(0, w - ww + 1, stride):
            windows_evaluated += 1
            crop = frame[y:y+wh, x:x+ww]
            score = ml_model.predict_crop(crop)
            if score >= threshold:
                boxes.append((x, y, ww, wh))
                scores.append(score)
                
    if not boxes:
        return [], 0
        
    final_boxes, final_scores = non_max_suppression(boxes, scores, iou_threshold=nms_iou_thresh)
    
    results = [(tuple(b), s) for b, s in zip(final_boxes, final_scores)]
    return results, windows_evaluated
