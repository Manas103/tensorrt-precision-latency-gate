#!/usr/bin/env python3
"""Evaluates a CameraOnlyDetector checkpoint on the held-out test range, against oracle
ground truth (not VLM labels): this is the only place in the pipeline oracle ground truth
is used for anything other than the label audit, because scoring a detector against the
same noisy labels it was trained on would not test anything.

Usage: python3 -m mining.evaluate --checkpoint mining/ckpt_mined.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mining.dataset import TEST_SEED_RANGE, add_sensor_noise, make_batch, to_chw
from mining.model import CameraOnlyDetector


def evaluate(checkpoint: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CameraOnlyDetector().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    start, end = TEST_SEED_RANGE
    seeds = np.arange(start, end)
    images_hwc, true_labels = make_batch(seeds)
    images_chw = to_chw(images_hwc)
    images_chw = add_sensor_noise(images_chw, seed=999_999)
    true_labels = np.array(true_labels)
    true_is_rare = (true_labels != "normal").astype(np.int64)

    x = torch.from_numpy(images_chw).float().to(device)
    with torch.no_grad():
        preds = []
        for b in range(0, x.shape[0], 512):
            logits = model(x[b:b + 512])
            preds.append(logits.argmax(dim=1).cpu().numpy())
    pred_is_rare = np.concatenate(preds)

    overall_accuracy = float((pred_is_rare == true_is_rare).mean())
    tp = int(((pred_is_rare == 1) & (true_is_rare == 1)).sum())
    fp = int(((pred_is_rare == 1) & (true_is_rare == 0)).sum())
    fn = int(((pred_is_rare == 0) & (true_is_rare == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    low_light_mask = true_labels == "low_light"
    low_light_recall = float(pred_is_rare[low_light_mask].mean()) if low_light_mask.any() else None

    per_class_recall = {}
    for cls in ("low_light", "wet_surface", "construction_zone"):
        mask = true_labels == cls
        per_class_recall[cls] = float(pred_is_rare[mask].mean()) if mask.any() else None

    return {
        "checkpoint": checkpoint,
        "n_test": int(len(seeds)),
        "overall_accuracy": overall_accuracy,
        "precision_rare": precision,
        "recall_rare": recall,
        "low_light_recall": low_light_recall,
        "per_class_recall": per_class_recall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
