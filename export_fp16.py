#!/usr/bin/env python3
"""Exports the trained checkpoint to an fp16 ONNX graph by calling
model.half() before torch.onnx.export, rather than exporting fp32 and
flagging the TensorRT builder for fp16: TensorRT 11.4 has no builder-level
FP16 flag any more (strong typing follows the graph's own dtypes, see
engines/build_engine.py), so the fp16-ness has to already be in the ONNX
file this script produces.
"""
import argparse

import torch

from model.cnn import SmallCIFARNet


def export_fp16_onnx(checkpoint: str, out: str) -> None:
    model = SmallCIFARNet().to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval().half()

    dummy = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float16)
    torch.onnx.export(
        model, dummy, out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--out", default="model_fp16.onnx")
    args = ap.parse_args()
    export_fp16_onnx(args.checkpoint, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
