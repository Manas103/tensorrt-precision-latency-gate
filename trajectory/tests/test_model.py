import torch

from trajectory.model import TrajectoryDecoder
from trajectory.scenario import N_HORIZON


def test_forward_pass_shape():
    model = TrajectoryDecoder(hidden=64)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, N_HORIZON * 2)


def test_different_hidden_sizes_are_different_parameter_counts():
    small = sum(p.numel() for p in TrajectoryDecoder(hidden=32).parameters())
    big = sum(p.numel() for p in TrajectoryDecoder(hidden=128).parameters())
    assert big > small
