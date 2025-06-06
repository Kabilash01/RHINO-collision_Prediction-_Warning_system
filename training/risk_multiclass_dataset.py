import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MultiClassRiskDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        print(f"[INFO] Loaded CSV with {len(self.df)} rows.")

        # Drop rows with any missing values
        self.df = self.df.dropna(subset=["vsv", "vlv", "headway", "label"])

        # Ensure label is integer
        self.df["label"] = self.df["label"].astype("int64")

        # Keep only rows with labels 0, 1, 2
        self.df = self.df[self.df["label"].isin([0, 1, 2])]

        self.features = self.df[["vsv", "vlv", "headway"]].values.astype("float32")
        self.labels = self.df["label"].values

        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"[INFO] Label distribution: {dict(zip(unique, counts))}")
        print(f"[INFO] Final dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y
