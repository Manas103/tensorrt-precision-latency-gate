#!/usr/bin/env python3
"""Produces the INT8 engine's input ONNX graph via explicit quantization
(QuantizeLinear / DequantizeLinear nodes inserted around the weights and
activations, with per-tensor scales computed from the calibration images),
using onnxruntime.quantization.quantize_static.

This replaces the original design, which called TensorRT's own
IInt8EntropyCalibrator2 implicit-calibration interface from build_engine.py
directly. TensorRT 11.4 (installed on this machine) does not expose that
class in its Python bindings at all (trt.IInt8EntropyCalibrator2 raises
AttributeError; confirmed empirically, and NVIDIA's own TensorRT 10.x-to-
11.x migration notes say implicit quantization is not supported in 11.x,
with explicit Q/DQ nodes as the only INT8 path). onnxruntime's static
quantizer is the standard, widely used tool for producing exactly that
Q/DQ graph from a calibration set; TensorRT's ONNX parser consumes the
resulting QuantizeLinear/DequantizeLinear nodes directly as explicit
quantization. This is a simpler per-tensor min/max-based calibration
(onnxruntime's default), not TensorRT's own KL-divergence entropy
calibration; the difference is disclosed, not hidden, in the README.
"""
import argparse

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class NpyCalibrationReader(CalibrationDataReader):
    def __init__(self, calib_npy: str, input_name: str, batch_size: int = 32):
        self.images = np.load(calib_npy).astype(np.float32)
        self.input_name = input_name
        self.batch_size = batch_size
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        batch = self.images[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return {self.input_name: batch}


def quantize(onnx_fp32_path: str, calib_npy: str, out_path: str, input_name: str = "input") -> None:
    reader = NpyCalibrationReader(calib_npy, input_name)
    quantize_static(
        onnx_fp32_path,
        out_path,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        # onnxruntime's default quantizes conv bias as int32 (zero-point
        # beyond int8 range, the standard QDQ convention for bias terms).
        # TensorRT 11.4's ONNX parser rejects that outright: "input has
        # type Int32 but must have type FP8, FP4, Int4, Int8, or UInt8"
        # on the bias DequantizeLinear node specifically. QuantizeBias=False
        # leaves bias in float and lets only weights and activations carry
        # the int8 QDQ nodes TensorRT 11.4 actually accepts.
        extra_options={
            "QuantizeBias": False,
            # TensorRT's ONNX parser rejects a non-zero zero-point outright
            # ("Non-zero zero point is not supported") outside DLA.
            # onnxruntime's int8 activation quantization is asymmetric by
            # default; forcing it symmetric keeps zero_point at exactly 0.
            "ActivationSymmetric": True,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--calib-npy", default="calib_images.npy")
    ap.add_argument("--out", default="model_int8_qdq.onnx")
    ap.add_argument("--input-name", default="input")
    args = ap.parse_args()
    quantize(args.onnx, args.calib_npy, args.out, input_name=args.input_name)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
