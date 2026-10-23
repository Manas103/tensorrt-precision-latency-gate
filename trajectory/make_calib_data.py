#!/usr/bin/env python3
"""Calibration images for INT8, drawn from the *train* seed range only,
same discipline as the repo-root make_calib_data.py: calibration data must
never overlap the range the accuracy budget is later measured on."""
import argparse

import numpy as np

from .dataset import make_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=512)
    ap.add_argument("--out", default="calib_bev.npy")
    args = ap.parse_args()

    images = np.stack([make_example(seed)[0] for seed in range(args.num_images)])
    np.save(args.out, images.astype(np.float32))
    print(f"wrote {args.out} shape={images.shape}")


if __name__ == "__main__":
    main()
