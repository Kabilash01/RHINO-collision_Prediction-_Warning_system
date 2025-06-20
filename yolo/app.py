import os
import sys
import cv2
import json
import socket
import torch
import numpy as np
import torch.nn as nn
from collections import deque
from ultralytics import YOLO
from accelerate import Accelerator

# === Paths Setup
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath("alerts"))

from velocity_model import VelocityPredictor
from visibility_classifier import predict_visibility_from_frame
from hybrid_headway import estimate_headway
from prediction_horizon import map_visibility_to_prt
from sms_alert import SmsAlert
from email_alert import EmailAlert

# === Risk Model
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

# === Initialize Models
accelerator = Accelerator()
device = accelerator.device
risk_model = RiskSequenceModel().to(device)
risk_model.load_state_dict(torch.load("models/risk_seq_model.pth", map_location=device))
risk_model.eval()

vsv_model = VelocityPredictor().to(device)
vsv_model.load_state_dict(torch.load("models/velocity_model.pth", map_location=device))
vsv_model.eval()

yolo = YOLO("yolov8n.pt")

# === Label Mapping
with open("label_mapping.json", "r") as f:
    label_map = json.load(f)
vehicle_ids = [int(k) for k, v in label_map.items() if v in ["car", "truck", "bus", "motorcycle", "bicycle"]]

# === Video Setup
cap = cv2.VideoCapture(0)
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Socket Setup (ESP32)
HOST = "192.168.4.1"  # ESP32 IP
PORT = 8080
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
print("[SOCKET] Connected to ESP32 at", HOST)

# === Runtime Vars
history = deque(maxlen=5)
last_speeds = [40.0, 42.0, 41.0]
RISK_THRESHOLDS = [0.7, 0.4, 0.2]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Get Sensor Data from ESP32
    try:
        esp_data = client.recv(1024).decode().strip()
        data = json.loads(esp_data)
        headway = float(data.get("distance", 200.0))
        humidity = float(data.get("humidity", 50.0))
        temp = float(data.get("temp", 30.0))
    except Exception as e:
        print("[WARN] ESP32 sensor data missing", e)
        continue

    # === YOLOv8 Detection
    results = yolo(frame)[0]
    boxes = results.boxes
    vehicle_boxes = [b for b in boxes if int(b.cls) in vehicle_ids]

    if vehicle_boxes:
        _, y1, _, y2 = vehicle_boxes[0].xyxy[0].cpu().numpy()
        box_height = y2 - y1
    else:
        box_height = 50.0

    # === VSV Prediction
    vsv_tensor = torch.tensor([last_speeds], dtype=torch.float32).to(device)
    vsv = vsv_model(vsv_tensor).item()
    last_speeds = [last_speeds[1], last_speeds[2], vsv]

    # === Dummy VLV
    vlv = 38.0

    # === TTC Calculation
    ttc = headway / (vsv - vlv + 1e-3)

    # === Forecasting
    history.append([vsv, vlv, headway])
    forecast = [0.0, 0.0, 0.0]
    if len(history) == 5:
        x_seq = torch.tensor(np.array(history), dtype=torch.float32).unsqueeze(0).to(device)
        forecast = risk_model(x_seq).squeeze(0).tolist()

    # === Visibility + PRT
    visibility = predict_visibility_from_frame(frame)
    prt = map_visibility_to_prt(visibility)

    # === Draw Risk Bars
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

    # === CRASH DETECTION
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

    # === Decision Logic & Alert
    if (forecast[0] > 0.3 and headway < 20.0 and ttc < prt + 1) or crash_flag:
        client.send(b"CRASH\n")
        SmsAlert("🚨 RHINO-X CRASH").run()
        EmailAlert("🚨 RHINO-X CRASH").run()
        cv2.putText(frame, "🔥 CRASH DETECTED!", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif forecast[0] > 0.15:
        client.send(b"RISK\n")
        cv2.putText(frame, "⚠️ RISK", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    else:
        client.send(b"SAFE\n")
        cv2.putText(frame, "✓ SAFE", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === Frame Info
    cv2.putText(frame, f"VSV: {vsv:.1f} VLV: {vlv:.1f} HW: {headway:.1f} TTC: {ttc:.1f}", (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Humidity: {humidity:.1f}% Temp: {temp:.1f}C", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 1)

    cv2.imshow("RHINO-X AI Safety System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
client.close()
cv2.destroyAllWindows()
