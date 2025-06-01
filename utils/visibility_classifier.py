import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# === Class mapping ===
CLASS_NAMES = ['sunny', 'cloudy', 'foggy', 'rainy']

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === Custom CNN model (same as training)
class WeatherClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)

# === Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeatherClassifier().to(device)
model.load_state_dict(torch.load("../models/weather_model.pth", map_location=device))
model.eval()

# === Inference function
def predict_visibility_from_frame(frame):
    image = Image.fromarray(frame).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()
        return CLASS_NAMES[pred]
