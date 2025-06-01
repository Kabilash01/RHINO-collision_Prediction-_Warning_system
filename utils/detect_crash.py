# detect_crash.py
import numpy as np

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def detect_crash(vehicle_boxes, previous_boxes, vsv, vlv, headway, risk_score):
    for box in vehicle_boxes:
        for prev in previous_boxes:
            iou = calculate_iou(box.xyxy[0].cpu().numpy(), prev.xyxy[0].cpu().numpy())
            if iou > 0.6 and risk_score > 0.9 and abs(vsv - vlv) > 30 and headway < 2.0:
                return True
    return False
