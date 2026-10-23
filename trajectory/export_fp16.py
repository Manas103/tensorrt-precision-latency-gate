#!/usr/bin/env python3
"""Exports a trained TrajectoryDecoder checkpoint to an fp16 ONNX graph,
the same reason and the same mechanism as the sibling repo-root
export_fp16.py: TensorRT 11.4's strongly-typed network mode reads
precision entirely from the ONNX graph's own dtypes, so the fp16-ness has
to already be in the exported graph, not a builder flag."""
import argparse

import torch

from .model import TrajectoryDecoder


def export_fp16_onnx(checkpoint: str, hidden: int, out: str) -> None:
    model = TrajectoryDecoder(hidden=hidden).to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval().half()

    dummy = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float16)
    torch.onnx.export(
        model, dummy, out,
        input_names=["bev"], output_names=["path"],
        dynamic_axes={"bev": {0: "batch"}, "path": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_c.pt")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", default="ckpt_c_fp16.onnx")
    args = ap.parse_args()
    export_fp16_onnx(args.checkpoint, args.hidden, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
