#!/usr/bin/env python3
"""Ranks the three checkpoints closed loop: each one drives the lightweight
rollout environment in closed_loop_env.py for 300 episodes (seeds disjoint
from every open-loop split) and is scored by collision rate + off-route
rate."""
import json

import torch

from .closed_loop_env import evaluate_closed_loop
from .model import TrajectoryDecoder
from .train import CHECKPOINTS

CLOSED_LOOP_SEED_BASE = 8_000_000
N_EPISODES = 300


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name, cfg in CHECKPOINTS.items():
        model = TrajectoryDecoder(hidden=cfg["hidden"]).to(device)
        model.load_state_dict(torch.load(f"ckpt_{name}.pt", map_location=device))
        results[name] = evaluate_closed_loop(model, device, N_EPISODES, CLOSED_LOOP_SEED_BASE)
        r = results[name]
        print(f"[{name}] collision_rate={r['collision_rate']:.3f} "
              f"off_route_rate={r['off_route_rate']:.3f} score={r['score']:.3f} "
              f"mean_closest_approach_m={r['mean_closest_approach_m']:.3f}")

    # Primary key: collision+off-route score (lower is better). Tiebreak:
    # mean closest-approach distance (higher is better, i.e. negated).
    ranking = sorted(results.keys(), key=lambda n: (results[n]["score"], -results[n]["mean_closest_approach_m"]))
    print(f"closed-loop ranking (best score first): {ranking}")

    with open("closed_loop_report.json", "w") as f:
        json.dump({"results": results, "ranking": ranking}, f, indent=2)
    print("wrote closed_loop_report.json")


if __name__ == "__main__":
    main()
