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

# === Add paths to utils and alerts folders ===
sys.path.append(os.path.abspath("../utils"))
sys.path.append(os.path.abspath("../alerts"))

# === Import project modules ===
from velocity_model import VelocityPredictor
from visibility_classifier import predict_visibility_from_frame
from hybrid_headway import estimate_headway
from prediction_horizon import map_visibility_to_prt
from vlv_tracker import get_vlv
from sms_alert import SmsAlert
from email_alert import EmailAlert

# === LSTM Risk Forecast Model ===
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

# === Accelerator for device optimization ===
accelerator = Accelerator()
device = accelerator.device

# === Load models ===
YOLO_MODEL = "yolov8n.pt"
LABEL_MAP_PATH = "../label_mapping.json"
risk_seq_model = RiskSequenceModel().to(device)
risk_seq_model.load_state_dict(torch.load("../models/risk_seq_model.pth", map_location=device))
risk_seq_model.eval()

vsv_model = VelocityPredictor().to(device)
vsv_model.load_state_dict(torch.load("../models/velocity_model.pth", map_location=device))
vsv_model.eval()

yolo = YOLO(YOLO_MODEL)

# === Load label map ===
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
vehicle_ids = [int(k) for k, v in label_map.items() if v in ["car", "truck", "bus", "motorcycle", "bicycle"]]

# === Wi-Fi socket setup ===
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"[📡] Waiting for ESP32 connection on {HOST}:{PORT}...")
conn, addr = sock.accept()
print(f"[✅] ESP32 connected from {addr}")

# === Runtime params ===
RISK_THRESHOLDS = [0.7, 0.4, 0.2]
history = deque(maxlen=5)
last_speeds = [40.0, 42.0, 41.0]

# === Main loop ===
while True:
    try:
        data = conn.recv(1024).decode().strip()
        if not data:
            continue
        try:
            sensor_data = json.loads(data)
            distance = float(sensor_data.get("distance", 200.0))
            humidity = float(sensor_data.get("humidity", 50.0))
            temperature = float(sensor_data.get("temperature", 30.0))
        except:
            print("[⚠] Invalid sensor data:", data)
            continue

        # === Process frame (optional if using webcam) ===
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[❌] Failed to capture frame")
            continue

        results = yolo(frame)[0]
        boxes = results.boxes
        vehicle_boxes = [b for b in boxes if int(b.cls) in vehicle_ids]

        visibility = predict_visibility_from_frame(frame)
        headway = estimate_headway(distance, vehicle_boxes[0].xyxy[0][3] - vehicle_boxes[0].xyxy[0][1]) if vehicle_boxes else distance

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

        print(f"[INFO] VSV={vsv:.2f} | VLV={vlv:.2f} | Headway={headway:.2f} | Forecast={forecast[0]:.3f} | TTC={ttc:.2f}")

        # === Decision logic
        if (forecast[0] > 0.3 and headway < 20.0 and ttc < prt + 1):
            print("🔥 CRASH DETECTED!")
            SmsAlert(location="⚠ Crash Detected Frame").run()
            EmailAlert(location="⚠ Crash Detected Frame").run()
            conn.sendall(b"CRASH\n")
        else:
            print("✓ SAFE")
            conn.sendall(b"SAFE\n")

    except KeyboardInterrupt:
        break

conn.close()
sock.close()
print("[✔] RHINO-X Shutdown Complete")
