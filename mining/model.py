"""CameraOnlyDetector: a small CNN binary classifier (rare scenario vs normal) over the
3x32x32 synthetic driving frame, the same encoder shape as the repo-root SmallCIFARNet and
trajectory.model.TrajectoryDecoder, so this repo's existing build_engine/EngineRunner/
quantize_int8 machinery applies unchanged (see mining/gate.py).
"""
import torch
import torch.nn as nn

NUM_CLASSES = 2  # 0 = normal, 1 = rare (low_light, wet_surface or construction_zone)


class CameraOnlyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.ReLU(),     # 32 -> 32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),   # 32 -> 16
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(),  # 16 -> 16
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),  # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
