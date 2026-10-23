"""Closed-loop safety check: the model predicts one 3-second path from the
t=0 BEV, exactly as it was trained to, and that path is checked against
every obstacle's own independent constant-speed motion over the same 3
seconds. This is deliberately single-shot (no iterative replanning): an
earlier version of this file replanned every 0.9s and tracked obstacles in
a continuously re-centered relative frame, which was complex enough that a
bookkeeping bug made the near-miss distance collapse to a two-valued
signal (see README Findings); single-shot removes that whole class of bug
while still scoring the thing the resume bullet means by "closed loop":
collision and off-route rate on the rolled-out path, not displacement
error against a label.

This environment does not import or call the CUDA simulator in the
sibling batched-gpu-driving-scenario-simulator repo; it is a standalone
environment written for this repo, consistent with every project in this
portfolio owning its own code.
"""
import numpy as np
import torch

from .scenario import DT_PLAN, N_HORIZON, VEHICLE_LEN, VEHICLE_WID, render_bev, sample_scenario

OFF_ROUTE_LAT_M = 5.4  # 1.5 lane widths: drifted off the 3-lane corridor entirely
SENSE_RANGE_M = 40.0   # matches the BEV's own forward range


def rollout_episode(model, device, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ego_lane, ego_speed, obstacles = sample_scenario(rng)
    bev = render_bev(ego_lane, ego_speed, obstacles)

    with torch.no_grad():
        x = torch.from_numpy(bev).unsqueeze(0).to(device)
        path = model(x).cpu().numpy().reshape(N_HORIZON, 2)

    collided = False
    off_route = False
    min_gap = SENSE_RANGE_M

    for k in range(N_HORIZON):
        fwd_k, lat_k = path[k]
        t = (k + 1) * DT_PLAN
        if abs(lat_k) > OFF_ROUTE_LAT_M:
            off_route = True
        for obs in obstacles:
            if obs.lane_idx != ego_lane:
                continue
            obs_relative_fwd_k = obs.forward_m + obs.speed * t - fwd_k
            if abs(obs_relative_fwd_k) > SENSE_RANGE_M:
                continue
            gap = abs(obs_relative_fwd_k) - VEHICLE_LEN
            min_gap = min(min_gap, gap)
            if gap < 0.0 and abs(lat_k) < VEHICLE_WID:
                collided = True

    return {"collided": collided, "off_route": off_route, "min_gap": min_gap}


def evaluate_closed_loop(model, device, n_episodes: int, seed_base: int) -> dict:
    """collision_rate and off_route_rate are the primary ranking key;
    mean_closest_approach_m (the average, over every episode, of the
    closest same-lane gap reached, clamped at 0 for episodes that
    collided) is a continuous tiebreak."""
    model.eval()
    n_collide = 0
    n_off_route = 0
    closest_approaches = []
    for i in range(n_episodes):
        result = rollout_episode(model, device, seed_base + i)
        n_collide += int(result["collided"])
        n_off_route += int(result["off_route"])
        closest_approaches.append(max(0.0, result["min_gap"]))
    return {
        "n_episodes": n_episodes,
        "collision_rate": n_collide / n_episodes,
        "off_route_rate": n_off_route / n_episodes,
        "score": (n_collide + n_off_route) / n_episodes,
        "mean_closest_approach_m": float(np.mean(closest_approaches)),
    }
