"""Builds the INT8 calibration set from a held-out slice of the CIFAR-10
*training* set, never the test set the gate's accuracy budget scores on
(see engines/calibrator.py's docstring for why that separation matters).
Writes a plain float32 NCHW .npy, already normalized the same way train.py
normalizes its inputs, since the calibrator feeds these straight to the
network with no further preprocessing.
"""
import argparse

import numpy as np
import torchvision
import torchvision.transforms as T

DATA_ROOT = "/tmp/cifar10"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=512)
    ap.add_argument("--out", default="calib_images.npy")
    args = ap.parse_args()

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=tf)

    # The last N images of the training set: train.py's loader shuffles
    # every epoch, so no fixed held-out split is baked into training, but
    # pulling from one fixed end of the dataset keeps this calibration
    # slice reproducible and separate from the test set regardless.
    n = args.num_images
    images = np.stack([train_set[i][0].numpy() for i in range(len(train_set) - n, len(train_set))])
    images = images.astype(np.float32)
    np.save(args.out, images)
    print(f"wrote {args.out}: {images.shape} {images.dtype}")


if __name__ == "__main__":
    main()
