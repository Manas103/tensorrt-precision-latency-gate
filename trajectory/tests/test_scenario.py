import numpy as np

from trajectory.scenario import (
    GRID_H, GRID_W, N_HORIZON, render_bev, rollout_rule_based_path, sample_scenario,
)


def test_bev_shape_and_range():
    rng = np.random.default_rng(1)
    ego_lane, ego_speed, obstacles = sample_scenario(rng)
    bev = render_bev(ego_lane, ego_speed, obstacles)
    assert bev.shape == (3, GRID_H, GRID_W)
    assert np.isfinite(bev).all()
    assert bev[0].sum() > 0  # lane boundaries are always drawn


def test_path_shape_and_determinism():
    rng1 = np.random.default_rng(42)
    ego_lane, ego_speed, obstacles = sample_scenario(rng1)
    path1 = rollout_rule_based_path(ego_lane, ego_speed, obstacles)
    rng2 = np.random.default_rng(42)
    ego_lane2, ego_speed2, obstacles2 = sample_scenario(rng2)
    path2 = rollout_rule_based_path(ego_lane2, ego_speed2, obstacles2)

    assert path1.shape == (N_HORIZON, 2)
    assert np.array_equal(path1, path2)


def test_ego_always_makes_forward_progress():
    """A stationary or reversing ego would mean the braking rule over-
    corrected; forward position must be non-decreasing every step."""
    for seed in range(50):
        rng = np.random.default_rng(seed)
        ego_lane, ego_speed, obstacles = sample_scenario(rng)
        path = rollout_rule_based_path(ego_lane, ego_speed, obstacles)
        fwd = path[:, 0]
        assert np.all(np.diff(fwd) >= -1e-9)


def test_no_obstacles_means_free_road_cruise():
    """With no obstacles the toy rule should accelerate toward the fixed
    cruise speed, not brake."""
    rng = np.random.default_rng(7)
    ego_lane, _, _ = sample_scenario(rng)
    path = rollout_rule_based_path(ego_lane, 10.0, [])
    speeds = np.diff(np.concatenate([[0.0], path[:, 0]])) / 0.3
    assert speeds[-1] > speeds[0]  # speed increases toward cruise
