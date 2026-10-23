#!/usr/bin/env python3
"""Trains three TrajectoryDecoder checkpoints (named a, b, c; different
hidden width and init seed, so they are genuinely different candidates, not
three copies of the same training run) on the synthetic BEV dataset. Not a
research training run: just enough that the three checkpoints separate on
both the open-loop and closed-loop rankings below, which is the
precondition for "the two rankings disagreed" to mean anything.
"""
import argparse
import time

import torch
import torch.nn as nn

from .dataset import TrajectoryDataset
from .model import TrajectoryDecoder

CHECKPOINTS = {
    "a": {"hidden": 32, "seed": 1},
    "b": {"hidden": 64, "seed": 2},
    "c": {"hidden": 128, "seed": 3},
}


def train_one(name: str, hidden: int, seed: int, epochs: int, device: str) -> TrajectoryDecoder:
    torch.manual_seed(seed)
    model = TrajectoryDecoder(hidden=hidden).to(device)

    train_set = TrajectoryDataset("train", n=4096)
    val_set = TrajectoryDataset("val", n=512)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.shape[0]
                n += x.shape[0]
        print(f"[{name}] epoch {epoch + 1}/{epochs} val_mse={val_loss / n:.4f} elapsed={time.time() - t0:.1f}s")

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, cfg in CHECKPOINTS.items():
        model = train_one(name, cfg["hidden"], cfg["seed"], args.epochs, device)
        model.eval()
        torch.save(model.state_dict(), f"ckpt_{name}.pt")

        dummy = torch.randn(1, 3, 32, 32, device=device)
        torch.onnx.export(
            model, dummy, f"ckpt_{name}.onnx",
            input_names=["bev"], output_names=["path"],
            dynamic_axes={"bev": {0: "batch"}, "path": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        print(f"saved ckpt_{name}.pt and ckpt_{name}.onnx (hidden={cfg['hidden']})")


if __name__ == "__main__":
    main()
