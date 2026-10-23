"""TrajectoryDecoder: a small CNN encoder over the 3x32x32 BEV raster and an
MLP head emitting 10 (forward, lateral) waypoints (3.0s at 0.3s spacing),
deliberately small so training and TensorRT compilation both take minutes,
not hours, on one GPU. `hidden` lets train.py produce checkpoints of
different capacity, which is also how the three ranked checkpoints differ
from each other.
"""
import torch
import torch.nn as nn

from .scenario import N_HORIZON


class TrajectoryDecoder(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),   # 32 -> 16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, N_HORIZON * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
