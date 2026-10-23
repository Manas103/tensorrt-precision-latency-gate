import numpy as np

from trajectory.metrics import average_displacement_error, final_displacement_error
from trajectory.scenario import N_HORIZON


def test_zero_error_when_identical():
    pred = np.random.default_rng(0).normal(size=(5, N_HORIZON * 2)).astype(np.float32)
    assert average_displacement_error(pred, pred) == 0.0
    assert final_displacement_error(pred, pred) == 0.0


def test_known_constant_offset():
    target = np.zeros((2, N_HORIZON, 2), dtype=np.float32)
    pred = target.copy()
    pred[..., 0] = 3.0
    pred[..., 1] = 4.0  # every waypoint off by (3, 4) -> distance 5
    ade = average_displacement_error(pred.reshape(2, -1), target.reshape(2, -1))
    fde = final_displacement_error(pred.reshape(2, -1), target.reshape(2, -1))
    assert abs(ade - 5.0) < 1e-6
    assert abs(fde - 5.0) < 1e-6


def test_fde_only_uses_last_waypoint():
    target = np.zeros((1, N_HORIZON, 2), dtype=np.float32)
    pred = target.copy()
    pred[0, 0, 0] = 100.0  # huge error on the first waypoint only
    fde = final_displacement_error(pred.reshape(1, -1), target.reshape(1, -1))
    assert fde == 0.0
