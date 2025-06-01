# utils/predict_headway_sequence.py
import torch
import numpy as np

MODEL_PATH = "models/headway_seq_model.pth"

class HeadwayPredictor:
    def __init__(self, model_class, seq_len=5, future_steps=3, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model_class(seq_len=seq_len, future_steps=future_steps).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        self.seq_len = seq_len
        self.future_steps = future_steps

    def predict(self, sequence):
        """
        sequence: list of shape [seq_len x 3] → [VSV, VLV, Headway]
        returns: predicted_headways: list of future_steps
        """
        with torch.no_grad():
            x = torch.tensor([sequence], dtype=torch.float32).to(self.device)
            out = self.model(x)
            return out.squeeze().tolist()
