import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperCnnBaseline(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        self.pool_final = nn.AdaptiveAvgPool1d(1)

        self.fc_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        x = self.pool_final(x)
        x = torch.flatten(x, 1)
        predictions = self.fc_head(x)

        return predictions
