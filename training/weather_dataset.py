import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class WeatherDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)["annotations"]  # <-- fixed

        self.img_dir = img_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.label_map = {
            "sunny": 0,
            "cloudy": 1,
            "foggy": 2,
            "rainy": 3
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = os.path.join(self.img_dir, os.path.basename(entry["filename"]))
        image = Image.open(img_path).convert("RGB")

        label_str = entry["weather"].lower()
        label = self.label_map.get(label_str, -1)

        return self.transform(image), label
