import torch
import torch.nn as nn

class PneumoniaCNNBaseline(nn.Module):
    def __init__(self, adapt_pool = 3, dropout_rate = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate)
        )
        self.pool = nn.AdaptiveAvgPool2d((adapt_pool, adapt_pool))
        self.hidden1 = nn.Linear(32*(adapt_pool**2),16)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
        self.output = nn.Linear(16,1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.output(x)
        return x
