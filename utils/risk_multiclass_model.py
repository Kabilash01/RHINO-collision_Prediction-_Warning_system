# risk_multiclass_model.py

import torch.nn as nn

class MulticlassRiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Output for 3 classes: Safe, Warning, Critical
        )

    def forward(self, x):
        return self.net(x)
