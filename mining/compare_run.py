#!/usr/bin/env python3
"""Trains both the mined and uniform strategies, evaluates both against oracle ground truth,
and reports the low-light recall gap directly: this is the number the resume's "14 points above
a uniformly sampled baseline" claim is measured against.

Usage: python3 -m mining.compare_run --scores mining/corpus_scores.npz --n-train 20000
"""
from __future__ import annotations

import argparse
import json

from mining.evaluate import evaluate
from mining.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="mining/corpus_scores.npz")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--out", default="mining/compare_report.json")
    args = ap.parse_args()

    train_mined = train("mined", args.scores, args.n_train, "mining/ckpt_mined.pt")
    train_uniform = train("uniform", args.scores, args.n_train, "mining/ckpt_uniform.pt")

    eval_mined = evaluate("mining/ckpt_mined.pt")
    eval_uniform = evaluate("mining/ckpt_uniform.pt")

    low_light_gap_points = (eval_mined["low_light_recall"] - eval_uniform["low_light_recall"]) * 100

    report = {
        "train_mined": {k: v for k, v in train_mined.items() if k != "loss_history"},
        "train_uniform": {k: v for k, v in train_uniform.items() if k != "loss_history"},
        "eval_mined": eval_mined,
        "eval_uniform": eval_uniform,
        "low_light_recall_gap_points": low_light_gap_points,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
