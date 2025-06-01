import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class HeadwaySequenceDataset(Dataset):
    def __init__(self, csv_file, seq_len=5, future_steps=3):
        df = pd.read_csv(csv_file)
        data = df[["VSV", "VLV", "Headway"]].values.astype(np.float32)
        targets = df["Headway"].values.astype(np.float32)

        self.X = []
        self.y = []

        for i in range(len(data) - seq_len - future_steps):
            self.X.append(data[i:i+seq_len])
            self.y.append(targets[i+seq_len:i+seq_len+future_steps])

        self.X = torch.tensor(np.array (self.X))
        self.y = torch.tensor( np.array(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
