#!/usr/bin/env python3
"""The trajectory-decoder analog of the repo-root gate.py: compiles the
best-ranked checkpoint (see rank_compare.py) to TensorRT engines at FP32,
FP16 and calibrated INT8, measures p50/p99 latency at batch 1 on this GPU,
and enforces a 1% relative-increase-in-displacement-error budget against
the FP32 PyTorch framework reference, the regression-output analog of the
root gate's 1% absolute top-1 accuracy budget. A precision that breaches
the budget is still reported, not silently dropped.
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

from .dataset import TrajectoryDataset
from .export_fp16 import export_fp16_onnx
from .metrics import average_displacement_error
from .model import TrajectoryDecoder

ACCURACY_BUDGET = 0.01  # relative increase in ADE vs the fp32 framework reference
N_WAYPOINTS_X2 = 20
N_TEST = 1024
N_LATENCY_REQUESTS = 300


def framework_reference(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    return {"pred": pred, "target": target, "ade": average_displacement_error(pred, target)}


def engine_eval(runner, loader, target_all):
    preds = []
    for x, _ in loader:
        x = x.to("cuda")
        out = runner.infer(x)
        preds.append(out.cpu().numpy())
    pred = np.concatenate(preds)
    return {"pred": pred, "ade": average_displacement_error(pred, target_all)}


def measure_latency_batch1(engine_path: str, n_requests: int) -> dict:
    runner = EngineRunner(engine_path, max_batch=1, num_classes=N_WAYPOINTS_X2)
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


def run_gate(checkpoint: str, hidden: int, onnx_path: str, calib_npy: str, out_json: str):
    device = "cuda"
    model = TrajectoryDecoder(hidden=hidden).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    test_set = TrajectoryDataset("test", n=N_TEST)
    loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    ref = framework_reference(model, loader, device)
    print(f"framework (fp32 PyTorch) ADE: {ref['ade']:.4f} m")

    fp16_onnx = "ckpt_best_fp16.onnx"
    export_fp16_onnx(checkpoint, hidden, fp16_onnx)
    int8_onnx = "ckpt_best_int8_qdq.onnx"
    quantize_int8(onnx_path, calib_npy, int8_onnx, input_name="bev")

    onnx_by_precision = {"fp32": onnx_path, "fp16": fp16_onnx, "int8": int8_onnx}
    engine_paths = {}
    for precision, path_in in onnx_by_precision.items():
        path = f"engine_{precision}.plan"
        build_engine(path_in, path, max_batch=64)
        engine_paths[precision] = path
        print(f"built {path} from {path_in}")

    report = {"framework_ade_m": ref["ade"], "accuracy_budget_relative": ACCURACY_BUDGET, "precisions": {}}

    for precision, path in engine_paths.items():
        runner = EngineRunner(path, max_batch=64, num_classes=N_WAYPOINTS_X2)
        result = engine_eval(runner, loader, ref["target"])
        latency = measure_latency_batch1(path, N_LATENCY_REQUESTS)

        rel_increase = (result["ade"] - ref["ade"]) / ref["ade"]
        blocked = rel_increase > ACCURACY_BUDGET

        report["precisions"][precision] = {
            "ade_m": result["ade"],
            "relative_ade_increase": rel_increase,
            "blocked": blocked,
            "latency": latency,
        }
        verdict = "BLOCKED (error budget breached)" if blocked else "passed"
        print(f"{precision}: ade={result['ade']:.4f}m rel_increase={rel_increase * 100:.2f}% "
              f"p50={latency['p50_ms']:.3f}ms p99={latency['p99_ms']:.3f}ms -> {verdict}")

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_json}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_c.pt")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--onnx", default="ckpt_c.onnx")
    ap.add_argument("--calib-npy", default="calib_bev.npy")
    ap.add_argument("--out", default="gate_report.json")
    args = ap.parse_args()
    run_gate(args.checkpoint, args.hidden, args.onnx, args.calib_npy, args.out)


if __name__ == "__main__":
    main()
