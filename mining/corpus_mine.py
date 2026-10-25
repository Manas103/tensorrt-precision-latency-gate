#!/usr/bin/env python3
"""Builds the 200,000-frame synthetic driving corpus and scores every frame with the CLIP VLM
scorer, writing per-frame true label, predicted label and predicted-rare probability to disk.
This is the "every frame scored with a vision-language model" claim, measured directly: the
output file has exactly CORPUS_SIZE rows, one VLM score per frame, no sampling.

Usage: python3 -m mining.corpus_mine --out mining/corpus_scores.npz
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from mining.dataset import CLASSES, CORPUS_SEED_RANGE, make_batch
from mining.vlm_scoring import ClipSceneScorer

BATCH_SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mining/corpus_scores.npz")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    start, end = CORPUS_SEED_RANGE
    seeds = np.arange(start, end)
    scorer = ClipSceneScorer(device="cuda")

    true_labels: list[str] = []
    pred_labels: list[str] = []
    rare_probs = np.zeros(len(seeds), dtype=np.float32)

    t0 = time.perf_counter()
    for batch_start in range(0, len(seeds), args.batch_size):
        batch_seeds = seeds[batch_start:batch_start + args.batch_size]
        images_hwc, labels = make_batch(batch_seeds)
        true_labels.extend(labels)
        images_uint8 = (images_hwc * 255).astype(np.uint8)
        probs = scorer.score_batch(images_uint8)
        pred_idx = probs.argmax(axis=1)
        pred_labels.extend(CLASSES[i] for i in pred_idx)
        normal_idx = CLASSES.index("normal")
        rare_probs[batch_start:batch_start + len(batch_seeds)] = 1.0 - probs[:, normal_idx]

    elapsed_s = time.perf_counter() - t0

    np.savez(
        args.out,
        seeds=seeds,
        true_labels=np.array(true_labels),
        pred_labels=np.array(pred_labels),
        rare_probs=rare_probs,
    )

    summary = {
        "num_frames_scored": len(seeds),
        "elapsed_s": round(elapsed_s, 2),
        "frames_per_sec": round(len(seeds) / elapsed_s, 1),
        "true_label_counts": {c: int((np.array(true_labels) == c).sum()) for c in CLASSES},
        "pred_label_counts": {c: int((np.array(pred_labels) == c).sum()) for c in CLASSES},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
