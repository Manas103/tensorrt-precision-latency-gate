#!/usr/bin/env python3
"""Calibration images for INT8, drawn from CALIB_SEED_RANGE only, which never overlaps the
training corpus (CORPUS_SEED_RANGE) or the held-out test range (TEST_SEED_RANGE): the same
discipline as the repo-root and trajectory/ make_calib_data.py."""
import argparse

import numpy as np

from mining.dataset import CALIB_SEED_RANGE, add_sensor_noise, make_batch, to_chw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mining/calib_frames.npy")
    args = ap.parse_args()

    start, end = CALIB_SEED_RANGE
    seeds = np.arange(start, end)
    images_hwc, _ = make_batch(seeds)
    images_chw = to_chw(images_hwc).astype(np.float32)
    images_chw = add_sensor_noise(images_chw, seed=888_888)
    np.save(args.out, images_chw)
    print(f"wrote {args.out} shape={images_chw.shape}")


if __name__ == "__main__":
    main()
