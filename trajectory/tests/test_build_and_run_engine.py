"""The trajectory-decoder analog of the repo-root
tests/test_build_and_run_engine.py: export a real (tiny) TrajectoryDecoder
to fp32, fp16 and a quantized INT8 QDQ graph, compile each to a real
TensorRT engine via the same engines/build_engine.py and engines/runner.py
every other precision in this repo uses, and diff against the PyTorch
model that was actually exported. This is the same shape of check gate.py
does at full scale on the best-ranked checkpoint.
"""
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("tensorrt")
pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from engines.build_engine import build_engine  # noqa: E402
from engines.runner import EngineRunner  # noqa: E402
from quantize_int8 import quantize  # noqa: E402

from trajectory.model import TrajectoryDecoder
from trajectory.scenario import N_HORIZON

NUM_OUT = N_HORIZON * 2


@pytest.fixture(scope="module")
def tiny_model_and_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("traj_tiny")
    model = TrajectoryDecoder(hidden=8).to("cuda").eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.01)  # same unscaled-fp16-overflow precaution as the root test
    return model, tmp


def test_fp32_engine_matches_pytorch_reference(tiny_model_and_dir):
    model, tmp = tiny_model_and_dir
    onnx_path = str(tmp / "tiny_fp32.onnx")
    dummy = torch.randn(1, 3, 32, 32, device="cuda")
    torch.onnx.export(model, dummy, onnx_path, input_names=["bev"], output_names=["path"],
                       dynamic_axes={"bev": {0: "batch"}, "path": {0: "batch"}}, opset_version=17, dynamo=False)

    engine_path = str(tmp / "tiny_fp32.plan")
    build_engine(onnx_path, engine_path)
    runner = EngineRunner(engine_path, max_batch=8, num_classes=NUM_OUT)

    x = torch.randn(8, 3, 32, 32, device="cuda")
    with torch.no_grad():
        expected = model(x)
    actual = runner.infer(x)
    assert (actual - expected).abs().max().item() < 1e-3


def test_fp16_engine_runs_fp16_end_to_end(tiny_model_and_dir):
    model, tmp = tiny_model_and_dir
    half_model = TrajectoryDecoder(hidden=8).to("cuda").eval()
    half_model.load_state_dict(model.state_dict())
    half_model.half()

    onnx_path = str(tmp / "tiny_fp16.onnx")
    dummy = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float16)
    torch.onnx.export(half_model, dummy, onnx_path, input_names=["bev"], output_names=["path"],
                       dynamic_axes={"bev": {0: "batch"}, "path": {0: "batch"}}, opset_version=17, dynamo=False)

    engine_path = str(tmp / "tiny_fp16.plan")
    build_engine(onnx_path, engine_path)
    runner = EngineRunner(engine_path, max_batch=8, num_classes=NUM_OUT)
    assert runner.input_dtype == torch.float16
    assert runner.output_dtype == torch.float16

    x = torch.randn(8, 3, 32, 32, device="cuda")
    with torch.no_grad():
        expected = model(x)
    actual = runner.infer(x)
    max_diff = (actual - expected).abs().max().item()
    assert max_diff < 0.2, f"fp16 engine diverged too far from the fp32 reference: {max_diff}"


def test_int8_qdq_engine_builds_and_runs(tiny_model_and_dir):
    model, tmp = tiny_model_and_dir
    onnx_path = str(tmp / "tiny_for_int8.onnx")
    dummy = torch.randn(1, 3, 32, 32, device="cuda")
    torch.onnx.export(model, dummy, onnx_path, input_names=["bev"], output_names=["path"],
                       dynamic_axes={"bev": {0: "batch"}, "path": {0: "batch"}}, opset_version=17, dynamo=False)

    calib_npy = str(tmp / "calib.npy")
    np.save(calib_npy, np.random.randn(32, 3, 32, 32).astype(np.float32))
    qdq_path = str(tmp / "tiny_int8_qdq.onnx")
    quantize(onnx_path, calib_npy, qdq_path, input_name="bev")

    engine_path = str(tmp / "tiny_int8.plan")
    build_engine(qdq_path, engine_path)
    runner = EngineRunner(engine_path, max_batch=8, num_classes=NUM_OUT)

    x = torch.randn(8, 3, 32, 32, device="cuda")
    out = runner.infer(x)
    assert out.shape == (8, NUM_OUT)
    assert torch.isfinite(out).all()
