"""Compiles an ONNX model to a TensorRT engine. This is a thin, explicit
wrapper around TensorRT's builder API (no trtexec), so every precision
decision is visible in one place.

TensorRT 11.4 (installed on this machine) removed implicit per-precision
builder flags entirely: trt.BuilderFlag has no FP16 or INT8 member, and
network creation has no EXPLICIT_BATCH flag either (confirmed empirically
and in NVIDIA's TensorRT 10.x-to-11.x migration guide: "TensorRT 11.x
removes the precision-enabling builder flags... Strong typing is always
on in 11.x"). Precision is now decided entirely by the dtypes already
present in the ONNX graph this function is handed: an fp32 ONNX graph
builds an fp32 engine, an fp16 ONNX graph (produced by exporting the
PyTorch model after model.half(), see gate.py) builds an fp16 engine, and
an INT8 engine comes from an ONNX graph with explicit QuantizeLinear /
DequantizeLinear (QDQ) nodes already inserted (see quantize_int8.py,
which uses onnxruntime.quantization.quantize_static for that, since
TensorRT's own IInt8EntropyCalibrator2 calibrator interface this project
originally called is not present in trt.11.4's Python bindings either).
This function itself is now precision-agnostic: it just builds whatever
the ONNX file in front of it says to build.
"""
import argparse

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, max_batch: int = 64) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    name = input_tensor.name
    shape = input_tensor.shape  # (-1, 3, 32, 32)
    fixed = (1,) + tuple(shape[1:])
    maxed = (max_batch,) + tuple(shape[1:])
    profile.set_shape(name, fixed, fixed, maxed)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()

    build_engine(args.onnx, args.out, max_batch=args.max_batch)
    print(f"built {args.out} from {args.onnx}")


if __name__ == "__main__":
    main()
