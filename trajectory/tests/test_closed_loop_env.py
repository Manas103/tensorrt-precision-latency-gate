import torch

from trajectory.closed_loop_env import evaluate_closed_loop, rollout_episode
from trajectory.model import TrajectoryDecoder


def test_rollout_episode_returns_expected_keys():
    model = TrajectoryDecoder(hidden=16)
    model.eval()
    result = rollout_episode(model, "cpu", seed=123)
    assert set(result.keys()) == {"collided", "off_route", "min_gap"}
    assert isinstance(result["collided"], bool)
    assert isinstance(result["off_route"], bool)


def test_evaluate_closed_loop_rates_are_in_unit_interval():
    model = TrajectoryDecoder(hidden=16)
    model.eval()
    report = evaluate_closed_loop(model, "cpu", n_episodes=20, seed_base=9_000_000)
    assert report["n_episodes"] == 20
    assert 0.0 <= report["collision_rate"] <= 1.0
    assert 0.0 <= report["off_route_rate"] <= 1.0


def test_same_seed_is_deterministic():
    model = TrajectoryDecoder(hidden=16)
    model.eval()
    r1 = rollout_episode(model, "cpu", seed=55)
    r2 = rollout_episode(model, "cpu", seed=55)
    assert r1 == r2
