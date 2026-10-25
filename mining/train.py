#!/usr/bin/env python3
"""Trains CameraOnlyDetector on a given seed set, using the VLM's predicted label as the
training target, not the oracle ground truth: this is the realistic version of the pipeline,
where the audit (label_audit.py) is what stands in for ground truth, not a guarantee that every
training label is correct. Evaluation (evaluate.py) is the only place oracle ground truth is used,
because that is also how a real deployment would be scored.

Usage: python3 -m mining.train --strategy mined --scores mining/corpus_scores.npz --n-train 20000 --out mining/ckpt_mined.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from mining.dataset import add_sensor_noise, make_batch, to_chw
from mining.model import CameraOnlyDetector
from mining.sampling import mined_training_seeds, uniform_training_seeds

EPOCHS = 8
BATCH_SIZE = 128
LR = 1e-3


def labels_for_seeds(scores_path: str, seeds: np.ndarray) -> np.ndarray:
    data = np.load(scores_path, allow_pickle=True)
    all_seeds = data["seeds"]
    pred_labels = data["pred_labels"]
    seed_to_idx = {int(s): i for i, s in enumerate(all_seeds)}
    idx = np.array([seed_to_idx[int(s)] for s in seeds])
    pred_is_rare = (pred_labels[idx] != "normal").astype(np.int64)
    return pred_is_rare


def train(strategy: str, scores_path: str, n_train: int, out_path: str, seed: int = 0) -> dict:
    if strategy == "mined":
        seeds = mined_training_seeds(scores_path, n_train)
    elif strategy == "uniform":
        seeds = uniform_training_seeds(scores_path, n_train)
    else:
        raise ValueError(strategy)

    train_labels = labels_for_seeds(scores_path, seeds)

    images_hwc, _ = make_batch(seeds)
    images_chw = to_chw(images_hwc)
    images_chw = add_sensor_noise(images_chw, seed=seed + 9000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(images_chw).float().to(device)
    y = torch.from_numpy(train_labels).long().to(device)

    torch.manual_seed(seed)
    model = CameraOnlyDetector().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    n = x.shape[0]
    history = []
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for b in range(0, n, BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        history.append(total_loss / n)

    torch.save(model.state_dict(), out_path)

    onnx_path = out_path.replace(".pt", ".onnx")
    model.eval()
    dummy = torch.randn(1, 3, 32, 32, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["frame"], output_names=["logits"],
        dynamic_axes={"frame": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    return {
        "strategy": strategy,
        "n_train": int(n),
        "rare_fraction_in_train": float(train_labels.mean()),
        "final_loss": history[-1],
        "loss_history": history,
        "checkpoint": out_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["mined", "uniform"], required=True)
    ap.add_argument("--scores", default="mining/corpus_scores.npz")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = train(args.strategy, args.scores, args.n_train, args.out)
    import json
    print(json.dumps({k: v for k, v in result.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
