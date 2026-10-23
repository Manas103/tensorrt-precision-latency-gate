"""Builds (BEV raster, flattened future-path) tensors from scenario.py, in
fixed, non-overlapping seed ranges per split so train/val/test never share
a scenario."""
import numpy as np
import torch

from .scenario import render_bev, rollout_rule_based_path, sample_scenario

SPLIT_SEED_BASE = {"train": 0, "val": 2_000_000, "test": 4_000_000}


def make_example(seed: int):
    rng = np.random.default_rng(seed)
    ego_lane, ego_speed, obstacles = sample_scenario(rng)
    bev = render_bev(ego_lane, ego_speed, obstacles)
    path = rollout_rule_based_path(ego_lane, ego_speed, obstacles)
    return bev, path


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, n: int):
        self.base = SPLIT_SEED_BASE[split]
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        bev, path = make_example(self.base + idx)
        return torch.from_numpy(bev), torch.from_numpy(path.reshape(-1))
