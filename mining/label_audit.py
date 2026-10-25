#!/usr/bin/env python3
"""Audits 1,000 of the VLM's "this frame is rare" labels against hand-checked ground truth,
before any of them are used to train the detector.

Honest framing: there is no human in this build. "Hand-checked" is a disclosed scripted proxy,
the ground-truth scenario label the synthetic generator assigned when it made the frame (see
mining/dataset.py docstring), the same precedent ground-truth-labeling-autolabeler-eval uses for
its scripted annotator. The audit is still a real, independent check: the VLM never sees this
ground truth, and the 1,000-frame sample is drawn after scoring, not selected to make the number
look good.

Usage: python3 -m mining.label_audit --scores mining/corpus_scores.npz --n 1000
"""
from __future__ import annotations

import argparse
import json

import numpy as np

AUDIT_SAMPLE_SEED = 1234


def audit(scores_path: str, n: int) -> dict:
    data = np.load(scores_path, allow_pickle=True)
    true_labels = data["true_labels"]
    pred_labels = data["pred_labels"]

    pred_is_rare = pred_labels != "normal"
    true_is_rare = true_labels != "normal"

    mined_indices = np.nonzero(pred_is_rare)[0]
    if len(mined_indices) < n:
        raise RuntimeError(f"only {len(mined_indices)} frames were predicted rare, need {n} to audit")

    rng = np.random.default_rng(AUDIT_SAMPLE_SEED)
    audit_indices = rng.choice(mined_indices, size=n, replace=False)

    audited_true_is_rare = true_is_rare[audit_indices]
    correct = int(audited_true_is_rare.sum())
    precision = correct / n

    per_class_in_audit = {
        cls: int((true_labels[audit_indices] == cls).sum())
        for cls in np.unique(true_labels)
    }

    return {
        "num_mined_frames_available": int(len(mined_indices)),
        "num_audited": n,
        "num_true_positive": correct,
        "precision": precision,
        "per_true_class_in_audit_sample": per_class_in_audit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="mining/corpus_scores.npz")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    result = audit(args.scores, args.n)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
