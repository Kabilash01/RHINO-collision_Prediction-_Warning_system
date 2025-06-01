import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from headway_seq_dataset import HeadwaySequenceDataset

# Config
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CSV_FILE = "C:/RHINO-CAR/training/real_gps_speed_data_ngsim.csv"
SEQ_LEN = 5
FUTURE_STEPS = 3

# Model
class HeadwayPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=FUTURE_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # use last time step output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = HeadwaySequenceDataset(CSV_FILE, seq_len=SEQ_LEN, future_steps=FUTURE_STEPS)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = HeadwayPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}] Batch {i} Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "headway_seq_model.pth")
    print("[✔] Model saved as headway_seq_model.pth")
