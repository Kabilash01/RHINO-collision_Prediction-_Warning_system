
import torch
import torch.nn as nn
import numpy as np

# === Model ===
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

# === Load model ===
model = RiskSequenceModel()
model.load_state_dict(torch.load("risk_seq_model.pth", map_location=torch.device("cpu")))
model.eval()

# === Sample input: last 5 frames of [VSV, VLV, Headway]
sample_input = np.array([
    [35.5, 33.1, 5.2],
    [36.0, 33.8, 5.1],
    [36.4, 34.2, 4.8],
    [36.8, 34.7, 4.5],
    [37.0, 35.0, 4.3]
], dtype=np.float32)

x_tensor = torch.tensor(sample_input).unsqueeze(0)  # shape: (1, 5, 3)

# === Predict future risk scores
with torch.no_grad():
    predicted_risk = model(x_tensor)
    print("Predicted Risk Levels (t+1 to t+3):", predicted_risk.numpy().flatten())
