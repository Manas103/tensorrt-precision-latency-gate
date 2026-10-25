#!/usr/bin/env python3
"""The mining extension's analog of the repo-root and trajectory/ gate.py: compiles the
deployed (mined-training) CameraOnlyDetector checkpoint to TensorRT engines at FP32, FP16 and
calibrated INT8, measures p50/p99 latency at batch 1 on this GPU, and enforces a 1% absolute
accuracy-drop budget against the FP32 PyTorch framework reference, scored against oracle ground
truth on the held-out test range (not VLM labels, see mining/evaluate.py docstring for why).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engines.build_engine import build_engine  # noqa: E402
from engines.runner import EngineRunner  # noqa: E402
from quantize_int8 import quantize as quantize_int8  # noqa: E402

from mining.dataset import TEST_SEED_RANGE, add_sensor_noise, make_batch, to_chw
from mining.export_fp16 import export_fp16_onnx
from mining.model import NUM_CLASSES, CameraOnlyDetector

ACCURACY_BUDGET = 0.01  # absolute accuracy drop vs the fp32 framework reference
N_LATENCY_REQUESTS = 300


def load_test_set():
    start, end = TEST_SEED_RANGE
    seeds = np.arange(start, end)
    images_hwc, true_labels = make_batch(seeds)
    images_chw = to_chw(images_hwc)
    images_chw = add_sensor_noise(images_chw, seed=999_999)
    true_is_rare = (np.array(true_labels) != "normal").astype(np.int64)
    return torch.from_numpy(images_chw).float(), torch.from_numpy(true_is_rare).long()


def framework_reference(model, x, y, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, x.shape[0], 512):
            logits = model(x[b:b + 512].to(device))
            preds.append(logits.argmax(dim=1).cpu())
    pred = torch.cat(preds)
    accuracy = (pred == y).float().mean().item()
    return {"pred": pred, "accuracy": accuracy}


def engine_eval(runner, x, y):
    preds = []
    for b in range(0, x.shape[0], 64):
        out = runner.infer(x[b:b + 64].to("cuda"))
        preds.append(out.argmax(dim=1).cpu())
    pred = torch.cat(preds)
    accuracy = (pred == y).float().mean().item()
    return {"pred": pred, "accuracy": accuracy}


def measure_latency_batch1(engine_path: str, n_requests: int) -> dict:
    runner = EngineRunner(engine_path, max_batch=1, num_classes=NUM_CLASSES)
    dummy = torch.randn(1, 3, 32, 32, device="cuda")

    runner.infer(dummy)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(n_requests):
        t0 = time.perf_counter()
        runner.infer(dummy)
        latencies.append(time.perf_counter() - t0)

    arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50) * 1000),
        "p99_ms": float(np.percentile(arr, 99) * 1000),
        "n_requests": n_requests,
        "batch_size": 1,
    }


def run_gate(checkpoint: str, onnx_path: str, calib_npy: str, out_json: str):
    device = "cuda"
    model = CameraOnlyDetector().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    x, y = load_test_set()
    ref = framework_reference(model, x, y, device)
    print(f"framework (fp32 PyTorch) accuracy: {ref['accuracy']:.4f}")

    fp16_onnx = "mining/ckpt_mined_fp16.onnx"
    export_fp16_onnx(checkpoint, fp16_onnx)
    int8_onnx = "mining/ckpt_mined_int8_qdq.onnx"
    quantize_int8(onnx_path, calib_npy, int8_onnx, input_name="frame")

    onnx_by_precision = {"fp32": onnx_path, "fp16": fp16_onnx, "int8": int8_onnx}
    engine_paths = {}
    for precision, path_in in onnx_by_precision.items():
        path = f"mining/engine_{precision}.plan"
        build_engine(path_in, path, max_batch=64)
        engine_paths[precision] = path
        print(f"built {path} from {path_in}")

    report = {"framework_accuracy": ref["accuracy"], "accuracy_budget": ACCURACY_BUDGET, "precisions": {}}

    for precision, path in engine_paths.items():
        runner = EngineRunner(path, max_batch=64, num_classes=NUM_CLASSES)
        result = engine_eval(runner, x, y)
        latency = measure_latency_batch1(path, N_LATENCY_REQUESTS)

        accuracy_drop = ref["accuracy"] - result["accuracy"]
        blocked = accuracy_drop > ACCURACY_BUDGET

        report["precisions"][precision] = {
            "accuracy": result["accuracy"],
            "accuracy_drop_vs_framework": accuracy_drop,
            "blocked": blocked,
            "latency": latency,
        }
        verdict = "BLOCKED (accuracy budget breached)" if blocked else "passed"
        print(f"{precision}: accuracy={result['accuracy']:.4f} drop={accuracy_drop:.4f} "
              f"p50={latency['p50_ms']:.4f}ms p99={latency['p99_ms']:.4f}ms -> {verdict}")

    fp32_p99 = report["precisions"]["fp32"]["latency"]["p99_ms"]
    for precision in ("fp16", "int8"):
        p99_speedup = fp32_p99 / report["precisions"][precision]["latency"]["p99_ms"]
        report["precisions"][precision]["p99_speedup_vs_fp32"] = p99_speedup
        print(f"{precision} p99 latency speedup vs fp32: {p99_speedup:.2f}x")

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_json}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="mining/ckpt_mined.pt")
    ap.add_argument("--onnx", default="mining/ckpt_mined.onnx")
    ap.add_argument("--calib-npy", default="mining/calib_frames.npy")
    ap.add_argument("--out", default="mining/gate_report.json")
    args = ap.parse_args()
    run_gate(args.checkpoint, args.onnx, args.calib_npy, args.out)


if __name__ == "__main__":
    main()
