"""Open-loop displacement-error metrics. pred and target are (B, N_HORIZON*2)
flattened (forward, lateral) waypoints."""
import numpy as np

from .scenario import N_HORIZON


def _reshape(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, N_HORIZON, 2)


def average_displacement_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean L2 distance between predicted and ground-truth waypoints,
    averaged over every waypoint and every example."""
    p, t = _reshape(pred), _reshape(target)
    dist = np.linalg.norm(p - t, axis=-1)
    return float(dist.mean())


def final_displacement_error(pred: np.ndarray, target: np.ndarray) -> float:
    """L2 distance at the last (3-second) waypoint only, averaged over
    examples."""
    p, t = _reshape(pred), _reshape(target)
    dist = np.linalg.norm(p[:, -1, :] - t[:, -1, :], axis=-1)
    return float(dist.mean())
