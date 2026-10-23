#!/usr/bin/env python3
"""Ranks the three checkpoints open loop: average displacement error (ADE)
against the ground-truth rule-based path over a held-out test split that
train.py never saw (the test seed range, see dataset.py)."""
import json

import numpy as np
import torch

from .dataset import TrajectoryDataset
from .metrics import average_displacement_error, final_displacement_error
from .train import CHECKPOINTS
from .model import TrajectoryDecoder


def evaluate_checkpoint(name: str, hidden: int, device: str, n_test: int = 1024) -> dict:
    model = TrajectoryDecoder(hidden=hidden).to(device)
    model.load_state_dict(torch.load(f"ckpt_{name}.pt", map_location=device))
    model.eval()

    test_set = TrajectoryDataset("test", n=n_test)
    loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(targets)

    return {
        "ade": average_displacement_error(pred, target),
        "fde": final_displacement_error(pred, target),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name, cfg in CHECKPOINTS.items():
        results[name] = evaluate_checkpoint(name, cfg["hidden"], device)
        print(f"[{name}] ade={results[name]['ade']:.3f}m fde={results[name]['fde']:.3f}m")

    ranking = sorted(results.keys(), key=lambda n: results[n]["ade"])
    print(f"open-loop ranking (best ADE first): {ranking}")

    with open("open_loop_report.json", "w") as f:
        json.dump({"results": results, "ranking": ranking}, f, indent=2)
    print("wrote open_loop_report.json")


if __name__ == "__main__":
    main()
