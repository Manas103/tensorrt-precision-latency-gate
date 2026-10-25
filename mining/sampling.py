"""Builds two training sets of equal size from the scored 200,000-frame corpus:
`mined`, drawn from the VLM's predicted-rare frames plus a matched number of
predicted-normal frames, and `uniform`, drawn uniformly at random from the whole
corpus with no regard to the VLM's prediction. Both are later trained on by the
identical model and training loop (see train.py); only the seed selection differs.
This isolates what mining buys: the comparison is mined-vs-uniform training data,
not mined-vs-uniform architecture or hyperparameters.
"""
from __future__ import annotations

import numpy as np

SAMPLING_SEED = 777


def mined_training_seeds(scores_path: str, n_total: int) -> np.ndarray:
    data = np.load(scores_path, allow_pickle=True)
    seeds = data["seeds"]
    pred_labels = data["pred_labels"]

    rare_mask = pred_labels != "normal"
    rare_seeds = seeds[rare_mask]
    normal_seeds = seeds[~rare_mask]

    n_rare = min(len(rare_seeds), n_total // 2)
    n_normal = n_total - n_rare

    rng = np.random.default_rng(SAMPLING_SEED)
    chosen_rare = rng.choice(rare_seeds, size=n_rare, replace=False)
    chosen_normal = rng.choice(normal_seeds, size=min(n_normal, len(normal_seeds)), replace=False)

    combined = np.concatenate([chosen_rare, chosen_normal])
    rng.shuffle(combined)
    return combined


def uniform_training_seeds(scores_path: str, n_total: int) -> np.ndarray:
    data = np.load(scores_path, allow_pickle=True)
    seeds = data["seeds"]
    rng = np.random.default_rng(SAMPLING_SEED + 1)
    return rng.choice(seeds, size=n_total, replace=False)
