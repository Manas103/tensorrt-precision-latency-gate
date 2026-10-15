#!/usr/bin/env python3
"""The regression gate: compiles one ONNX model to TensorRT engines at
FP32, FP16 and calibrated INT8, measures p50/p99 latency at a fixed batch
size and concurrency on this one GPU, diffs each engine's per-class logits
against the PyTorch (framework) reference, and enforces a 1% absolute
top-1 accuracy budget against that same reference: a precision that
breaches the budget is blocked regardless of how much faster it is.
"""
import argparse
import json
import threading
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from model.cnn import SmallCIFARNet
from engines.build_engine import build_engine
from engines.runner import EngineRunner
from export_fp16 import export_fp16_onnx
from quantize_int8 import quantize as quantize_int8

DATA_ROOT = "/tmp/cifar10"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

ACCURACY_BUDGET = 0.01  # absolute top-1 drop vs the fp32 framework reference
BATCH_SIZE = 32
CONCURRENCY = 4
REQUESTS_PER_WORKER = 150


def get_test_loader(batch_size=BATCH_SIZE):
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)


def framework_reference(model: torch.nn.Module, loader) -> dict:
    """Runs the PyTorch fp32 model on the full test set once. This is the
    framework reference every engine's accuracy and per-class logits are
    diffed against, not a second TensorRT engine."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to("cuda")
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    top1 = (logits.argmax(dim=1) == labels).float().mean().item()
    return {"logits": logits, "labels": labels, "top1": top1}


def engine_eval(runner: EngineRunner, loader) -> dict:
    """Runs one TensorRT engine over the full test set, diffed against
    nothing yet (the caller compares these logits to the framework
    reference's); returns this engine's own logits, labels and top1."""
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to("cuda")
        out = runner.infer(x)
        all_logits.append(out.cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    top1 = (logits.argmax(dim=1) == labels).float().mean().item()
    return {"logits": logits, "labels": labels, "top1": top1}


def per_class_logit_diff(engine_logits: torch.Tensor, framework_logits: torch.Tensor, num_classes: int = 10) -> dict:
    """Diffs each engine's per-class output column against the framework
    reference's, not just the top1 decision: a precision can flip very few
    decisions while still drifting substantially on every logit."""
    diff = (engine_logits - framework_logits).abs()
    per_class_max = diff.max(dim=0).values.tolist()
    per_class_mean = diff.mean(dim=0).tolist()
    return {
        "max_abs_diff_overall": diff.max().item(),
        "mean_abs_diff_overall": diff.mean().item(),
        "per_class_max_abs_diff": per_class_max,
        "per_class_mean_abs_diff": per_class_mean,
    }


def measure_latency(engine_path: str, max_batch: int, concurrency: int, batch_size: int, requests_per_worker: int) -> dict:
    """concurrency independent EngineRunner instances, each its own
    deserialized execution context and CUDA stream against the same
    engine file, each firing requests_per_worker sequential calls in its
    own thread. Latency is measured per call, end to end (host submit to
    host-visible result), on a torch CUDA stream synced after every call,
    which is what a real request actually waits on."""
    runners = [EngineRunner(engine_path, max_batch=max_batch) for _ in range(concurrency)]
    latencies_per_worker = [[] for _ in range(concurrency)]
    dummy_batch = torch.randn(batch_size, 3, 32, 32, device="cuda")

    # Warm up every context once outside the timed region: the first call
    # on a fresh execution context pays one-time CUDA context/stream setup
    # cost that a real serving process would also only pay once.
    for r in runners:
        r.infer(dummy_batch)
    torch.cuda.synchronize()

    def worker(idx):
        r = runners[idx]
        for _ in range(requests_per_worker):
            t0 = time.perf_counter()
            r.infer(dummy_batch)
            latencies_per_worker[idx].append(time.perf_counter() - t0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    all_latencies = np.array([lat for worker_lats in latencies_per_worker for lat in worker_lats])
    total_requests = len(all_latencies)
    throughput = (total_requests * batch_size) / wall
    return {
        "p50_ms": float(np.percentile(all_latencies, 50) * 1000),
        "p99_ms": float(np.percentile(all_latencies, 99) * 1000),
        "throughput_images_per_sec": throughput,
        "total_requests": total_requests,
        "wall_seconds": wall,
        "batch_size": batch_size,
        "concurrency": concurrency,
    }


def run_gate(checkpoint: str, onnx_path: str, calib_npy: str, out_json: str):
    model = SmallCIFARNet().to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval()

    loader = get_test_loader()
    ref = framework_reference(model, loader)
    print(f"framework (fp32 PyTorch) top1: {ref['top1']:.4f}")

    fp16_onnx = "model_fp16.onnx"
    export_fp16_onnx(checkpoint, fp16_onnx)
    int8_onnx = "model_int8_qdq.onnx"
    quantize_int8(onnx_path, calib_npy, int8_onnx)

    onnx_by_precision = {"fp32": onnx_path, "fp16": fp16_onnx, "int8": int8_onnx}
    engine_paths = {}
    for precision, path_in in onnx_by_precision.items():
        path = f"engine_{precision}.plan"
        build_engine(path_in, path)
        engine_paths[precision] = path
        print(f"built {path} from {path_in}")

    report = {"framework_top1": ref["top1"], "accuracy_budget": ACCURACY_BUDGET, "precisions": {}}

    for precision, path in engine_paths.items():
        runner = EngineRunner(path, max_batch=BATCH_SIZE)
        result = engine_eval(runner, loader)
        diff = per_class_logit_diff(result["logits"], ref["logits"])
        latency = measure_latency(path, max_batch=BATCH_SIZE, concurrency=CONCURRENCY,
                                   batch_size=BATCH_SIZE, requests_per_worker=REQUESTS_PER_WORKER)

        accuracy_drop = ref["top1"] - result["top1"]
        blocked = accuracy_drop > ACCURACY_BUDGET

        report["precisions"][precision] = {
            "top1": result["top1"],
            "accuracy_drop_vs_framework": accuracy_drop,
            "blocked": blocked,
            "logit_diff": diff,
            "latency": latency,
        }
        verdict = "BLOCKED (accuracy budget breached)" if blocked else "passed"
        print(f"{precision}: top1={result['top1']:.4f} drop={accuracy_drop:.4f} "
              f"p50={latency['p50_ms']:.3f}ms p99={latency['p99_ms']:.3f}ms "
              f"throughput={latency['throughput_images_per_sec']:.0f} img/s -> {verdict}")

    fp32_throughput = report["precisions"]["fp32"]["latency"]["throughput_images_per_sec"]
    for precision in ("fp16", "int8"):
        speedup = report["precisions"][precision]["latency"]["throughput_images_per_sec"] / fp32_throughput
        report["precisions"][precision]["speedup_vs_fp32"] = speedup
        print(f"{precision} throughput speedup vs fp32: {speedup:.2f}x")

    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_json}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--calib-npy", default="calib_images.npy")
    ap.add_argument("--out", default="gate_report.json")
    args = ap.parse_args()
    run_gate(args.checkpoint, args.onnx, args.calib_npy, args.out)


if __name__ == "__main__":
    main()
