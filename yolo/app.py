import os
import sys
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
from collections import deque
from ultralytics import YOLO
from accelerate import Accelerator

# RHINO-X Utility Imports
sys.path.append(os.path.abspath("../utils"))
sys.path.append(os.path.abspath("../alerts"))

from velocity_model import VelocityPredictor
from visibility_classifier import predict_visibility_from_frame
from hybrid_headway import estimate_headway
from prediction_horizon import map_visibility_to_prt
from vlv_tracker import get_vlv
from sms_alert import SmsAlert
from email_alert import EmailAlert

# Accelerator for mixed device training/inference
accelerator = Accelerator()
device = accelerator.device

# Risk Sequence Model
class RiskSequenceModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# === Paths ===
YOLO_MODEL = "yolov8n.pt"
VIDEO_PATH = "C:/RHINO-CAR/test_videos/test (6).mp4"
OUTPUT_PATH = "../output_video/forecasted_risk_output.mp4"
LABEL_MAP_PATH = "../label_mapping.json"

# === Load Models ===
risk_seq_model = RiskSequenceModel().to(device)
risk_seq_model.load_state_dict(torch.load("../models/risk_seq_model.pth", map_location=device))
risk_seq_model.eval()

vsv_model = VelocityPredictor().to(device)
vsv_model.load_state_dict(torch.load("../models/velocity_model.pth", map_location=device))
vsv_model.eval()

yolo = YOLO(YOLO_MODEL)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
vehicle_ids = [int(k) for k, v in label_map.items() if v in ["car", "truck", "bus", "motorcycle", "bicycle"]]

cap = cv2.VideoCapture(VIDEO_PATH)
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

DEFAULT_SENSOR = 250.0
RISK_THRESHOLDS = [0.7, 0.4, 0.2]
last_speeds = [40.0, 42.0, 41.0]
history = deque(maxlen=5)

print("[INFO] Starting RHINO-X video analysis...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]
    boxes = results.boxes
    vehicle_boxes = [b for b in boxes if int(b.cls) in vehicle_ids]

    visibility = predict_visibility_from_frame(frame)
    sensor_value = DEFAULT_SENSOR

    _, y1, _, y2 = vehicle_boxes[0].xyxy[0] if vehicle_boxes else (0, 0, 0, 0)
    box_height = y2 - y1
    headway = estimate_headway(sensor_value, box_height)

    vsv_tensor = torch.tensor([last_speeds], dtype=torch.float32).to(device)
    vsv = vsv_model(vsv_tensor).item()
    last_speeds = [last_speeds[1], last_speeds[2], vsv]
    vlv = get_vlv()

    history.append([vsv, vlv, headway])
    forecast = [0.0, 0.0, 0.0]
    if len(history) == 5:
        seq_input = torch.tensor(np.array(history), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            forecast = risk_seq_model(seq_input).squeeze(0).tolist()

    prt = map_visibility_to_prt(visibility)
    ttc = headway / (vsv - vlv + 1e-3)
    print(f"[DEBUG] forecast={forecast[0]:.3f}, headway={headway:.2f}, ttc={ttc:.2f}, prt={prt:.2f}")

    # === Forecast Bars
    for i, risk in enumerate(forecast):
        color = (0, 255, 0)
        if risk > RISK_THRESHOLDS[0]:
            color = (0, 0, 255)
        elif risk > RISK_THRESHOLDS[1]:
            color = (0, 255, 255)
        elif risk > RISK_THRESHOLDS[2]:
            color = (0, 165, 255)
        x = 30 + i * 50
        cv2.rectangle(frame, (x, 40), (x + 30, 70), color, -1)
        cv2.putText(frame, f"{risk:.2f}", (x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # === Crash Detection (IOU)
    crash_flag = False
    if len(vehicle_boxes) >= 2:
        box1 = vehicle_boxes[0].xyxy[0].cpu().numpy()
        box2 = vehicle_boxes[1].xyxy[0].cpu().numpy()
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = interArea / (boxAArea + boxBArea - interArea + 1e-3)
        if iou > 0.3:
            crash_flag = True
            print(f"[💥] Overlap CRASH DETECTED! (IoU={iou:.2f})")

    # === Alerts
    if (forecast[0] > 0.3 and headway < 20.0 and ttc < prt + 1) or crash_flag:
        cv2.putText(frame, "🔥 CRASH DETECTED!", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        SmsAlert(location="⚠ Crash Detected Frame").run()
        EmailAlert(location="⚠ Crash Detected Frame").run()
    else:
        cv2.putText(frame, "✓ SAFE", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === Annotations
    cv2.putText(frame, f"VSV: {vsv:.1f} VLV: {vlv:.1f} Headway: {headway:.1f}", (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Weather: {visibility}", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 1)

    out.write(cv2.resize(frame, (fw, fh)))
    cv2.imshow("RHINO-X Forecasting + Crash Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[✔] Video saved at: {OUTPUT_PATH}")
