import torch

from mining.model import NUM_CLASSES, CameraOnlyDetector


def test_forward_pass_shape():
    model = CameraOnlyDetector()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_forward_pass_batch_one():
    model = CameraOnlyDetector()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES)
