#!/usr/bin/env python3
"""Exports a trained CameraOnlyDetector checkpoint to an fp16 ONNX graph, the same reason and
mechanism as the repo-root and trajectory/ export_fp16.py: TensorRT 11.4's strongly-typed network
mode reads precision from the ONNX graph's own dtypes, not a builder flag."""
import argparse

import torch

from mining.model import CameraOnlyDetector


def export_fp16_onnx(checkpoint: str, out: str) -> None:
    model = CameraOnlyDetector().to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval().half()

    dummy = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float16)
    torch.onnx.export(
        model, dummy, out,
        input_names=["frame"], output_names=["logits"],
        dynamic_axes={"frame": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="mining/ckpt_mined.pt")
    ap.add_argument("--out", default="mining/ckpt_mined_fp16.onnx")
    args = ap.parse_args()
    export_fp16_onnx(args.checkpoint, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
