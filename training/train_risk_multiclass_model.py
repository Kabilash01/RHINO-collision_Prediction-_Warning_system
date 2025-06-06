import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from risk_multiclass_dataset import MultiClassRiskDataset

# === Config ===
EPOCHS = 20  # You can increase if needed
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CSV_FILE = "C:/RHINO-CAR/training/risk_multiclass_synthetic.csv"

# === Model ===
class MultiClassRiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3 output classes
        )

    def forward(self, x):
        return self.net(x)

# === Training ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = MultiClassRiskDataset(CSV_FILE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultiClassRiskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for i, (xb, yb) in enumerate(data_loader):
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 🟢 Print batch-wise loss like your screenshot
            sample_num = (i + 1) * BATCH_SIZE
            print(f"[Epoch {epoch+1}] Batch {sample_num} Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "risk_multiclass_model.pth")
    print("[✔] Model saved as risk_multiclass_model.pth")
