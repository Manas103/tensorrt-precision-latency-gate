import torch

from model.cnn import SmallCIFARNet


def test_forward_pass_shape():
    model = SmallCIFARNet(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_dynamic_batch_size():
    model = SmallCIFARNet(num_classes=10)
    for batch in (1, 3, 32):
        out = model(torch.randn(batch, 3, 32, 32))
        assert out.shape == (batch, 10)


def test_parameter_count_is_small_enough_to_train_in_minutes():
    model = SmallCIFARNet()
    n_params = sum(p.numel() for p in model.parameters())
    # Sanity bound: this is meant to be a small, fast-to-compile network,
    # not a bug check on an exact number that would break on any edit.
    assert 50_000 < n_params < 2_000_000
