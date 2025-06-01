import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from weather_dataset import WeatherDataset

# Configuration
JSON_FILE = "train.json"
IMG_DIR = "train_images"
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on: {device}")

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset = WeatherDataset(JSON_FILE, IMG_DIR, transform)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Model (Pretrained ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 weather classes
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == yb).sum().item()

    acc = correct / len(train_ds)
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_correct += (preds.argmax(1) == yb).sum().item()

    val_acc = val_correct / len(val_ds)
    print(f"           Val Accuracy: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "weather_model.pth")
print("[✔] Model saved as weather_model.pth")
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from weather_dataset import WeatherDataset

# Configuration
JSON_FILE = "train.json"
IMG_DIR = "train_images"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on: {device}")

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset = WeatherDataset(JSON_FILE, IMG_DIR, transform)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Model (Pretrained ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 weather classes
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == yb).sum().item()

    acc = correct / len(train_ds)
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_correct += (preds.argmax(1) == yb).sum().item()

    val_acc = val_correct / len(val_ds)
    print(f"           Val Accuracy: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "weather_model.pth")
print("[✔] Model saved as weather_model.pth")
