# TensorRT Precision and Latency Regression Gate

A CI-shaped gate that compiles one ONNX image classifier to TensorRT
engines at FP32, FP16 and INT8, measures p50/p99 latency at a fixed batch
size and concurrency on one GPU, diffs every engine's per-class logits
against the PyTorch framework reference, and enforces a 1% absolute top-1
accuracy budget: a precision that breaches the budget is blocked
regardless of how much faster it is. Every number below was measured on
this machine via `python gate.py`, not targeted.

## Why this exists

INT8 is always faster. The question that actually matters in production is
whether it is faster *and* correct enough to ship, and that is a judgment
a build pipeline should make automatically rather than leaving it to
whoever happens to eyeball an accuracy number before a release. This gate
makes that call the same way every time: compile all three precisions,
measure the same accuracy metric the same way for each, and block
anything past budget, independent of its speed.

## What this is NOT

- **Not TensorRT's implicit-calibration INT8 path, because that path does
  not exist any more on this machine's TensorRT.** This project was
  designed around `trt.IInt8EntropyCalibrator2` and `BuilderFlag.INT8`/
  `BuilderFlag.FP16`. TensorRT 11.4 (installed here) removed all of those:
  `trt.IInt8EntropyCalibrator2` is not present in the Python bindings at
  all (confirmed empirically: `AttributeError`), and `trt.BuilderFlag` has
  no `FP16` or `INT8` member (confirmed by listing it; NVIDIA's own
  TensorRT 10.x-to-11.x migration guide [1] says the same: strong typing
  is always on in 11.x, and implicit quantization is not supported). See
  Findings for what this project does instead.
- **Not trtexec.** Every build step is explicit Python against TensorRT's
  builder API, so every precision decision is visible in one place
  (`engines/build_engine.py`), not hidden in a CLI flag.
- **Not a research model.** `model/cnn.py` is a small CIFAR-10 CNN, deep
  enough to have real accuracy worth losing (the precondition for the
  accuracy-budget gate to mean anything) and small enough to train and
  compile in minutes on one GPU.
- **INT8 calibration here is a simpler per-tensor min/max scale, not
  entropy calibration.** `quantize_int8.py` uses
  `onnxruntime.quantization.quantize_static`'s default calibration method,
  not TensorRT's KL-divergence entropy calibrator (which, again, is not
  available to call on this TensorRT version). This is disclosed, not
  hidden, and it is a real candidate explanation if INT8's accuracy drop
  comes out worse than a hand-tuned entropy calibration would give.
- **"Concurrency" is N independent TensorRT execution contexts on the same
  engine, driven by N Python threads, not a production serving stack
  (no batching queue, no dynamic batching, no Triton Inference Server).**

## Machine and toolchain

WSL2 Ubuntu 22.04 on Windows 11, NVIDIA GeForce RTX 4070 SUPER (sm_89),
PyTorch 2.11.0+cu128, TensorRT 11.4.0.106, ONNX 1.23.2, onnxruntime
(static quantization only), Python 3.10.12.

## Architecture

```
model/cnn.py             SmallCIFARNet: a small CIFAR-10 CNN (6 conv layers + linear head)
train.py                 trains SmallCIFARNet on real CIFAR-10, exports model.onnx (fp32)
make_calib_data.py       held-out slice of the *training* set for INT8 calibration, never the test set
export_fp16.py           exports model.half() to an fp16 ONNX graph (TensorRT 11.4 needs this; see Findings)
quantize_int8.py         onnxruntime static quantization -> an INT8 QDQ ONNX graph
engines/
  build_engine.py          parses an ONNX graph, builds a strongly-typed TensorRT engine
  runner.py                 runs an engine via the tensor-address execution API, dtype-aware
gate.py                  orchestrates all of the above into the actual pass/block decision
tests/                   pytest suite (model, gate decision logic, engine build/run reference oracle)
docs/                    committed raw gate and test output
```

### Design deep-dives

**Why FP16 and INT8 are separate ONNX graphs, not builder flags.** TensorRT 11.4's strongly-typed network mode means the engine's precision comes entirely from the dtypes already in the graph the builder is handed (see "What this is NOT"). `export_fp16.py` produces a graph that is fp16 end to end by calling `model.half()` before `torch.onnx.export`; `quantize_int8.py` produces a graph with explicit `QuantizeLinear`/`DequantizeLinear` nodes around the quantized tensors. `engines/build_engine.py` itself does not know or care which precision it is building; it just compiles whatever graph it is given.

**Why `EngineRunner` reads tensor dtypes from the engine instead of assuming float32.** The fp16 engine's input and output tensor addresses must point at fp16-sized buffers, or `set_tensor_address` reads and writes past what a float32 buffer actually holds. `runner.py` queries `engine.get_tensor_dtype(...)` for both the input and output binding and allocates accordingly, converting the caller's input to match and always returning float32 logits, so every caller can compare across precisions without a dtype mismatch becoming its own source of spurious diff.

**Why calibration data comes from the training set, not the test set.** `make_calib_data.py`'s docstring states this directly: calibration data that overlaps the accuracy-budget's evaluation set would let INT8's quantization scales be implicitly tuned to the exact data the gate later scores it on, which is not a mistake this gate is supposed to be able to make even by accident.

**Why latency is measured with independent execution contexts, not one shared context from multiple threads.** A single `IExecutionContext` is not safe to call concurrently from multiple threads; the standard pattern for N-way concurrent serving from one engine is N execution contexts, each with its own CUDA stream. `gate.py`'s `measure_latency` creates exactly that many `EngineRunner` instances against the same engine file (each `EngineRunner.__init__` deserializes its own context), and drives them from N Python threads, each firing `requests_per_worker` sequential calls and recording per-call latency.

## Validation

**Unit tests (`tests/test_model.py`, `tests/test_gate_logic.py`).** Shape and parameter-count sanity on the model; the accuracy-budget decision rule (`accuracy_drop > ACCURACY_BUDGET`) and the per-class logit diff function are tested directly against hand-built tensors, including a corrupted-single-class case that confirms the diff isolates the right column rather than just reporting a global number.

**Reference-oracle engine tests (`tests/test_build_and_run_engine.py`).** A tiny model, independent of SmallCIFARNet, is exported at fp32 and fp16 and quantized to an INT8 QDQ graph, each compiled to a real TensorRT engine and run, diffed against the same PyTorch model in plain eager mode. This is the same shape of check `gate.py` does on the full CIFAR-10 model, kept small so the suite does not need the dataset download or a trained checkpoint to validate the build/run/quantize path itself.

```
python -m pytest tests/ -v
```

Raw output: `docs/test_output.txt`.

## Findings

**TensorRT 11.4 removed the exact calibration and precision-flag APIs this project was designed around, discovered by actually trying to run it, not by reading a changelog first.** The original build called `trt.IInt8EntropyCalibrator2` directly from an INT8-calibrator class and set `trt.BuilderFlag.FP16`/`trt.BuilderFlag.INT8` on the builder config. Both failed immediately on this machine: `AttributeError: module 'tensorrt' has no attribute 'IInt8EntropyCalibrator2'`, and `trt.BuilderFlag` has no `FP16` or `INT8` member at all (confirmed by listing `dir(trt.BuilderFlag)`: only `DEBUG`, `DIRECT_IO`, `GPU_FALLBACK`, `REFIT`, `SPARSE_WEIGHTS`, `STRICT_NANS`, `TF32`, `VERSION_COMPATIBLE`, `WEIGHT_STREAMING` and a few others remain; no precision flags survive). NVIDIA's own TensorRT 10.x-to-11.x migration guide [1] confirms this was deliberate: strong typing is always on in 11.x, and implicit quantization (the calibrator-based INT8 path) is not supported; the replacement is explicit QuantizeLinear/DequantizeLinear nodes in the graph. The fix: `export_fp16.py` builds the fp16 ONNX graph PyTorch-side instead of flagging the builder, and `quantize_int8.py` uses `onnxruntime.quantization.quantize_static` to produce the INT8 QDQ graph, since onnxruntime's quantizer is a standard tool for exactly this and TensorRT's ONNX parser consumes its QDQ nodes directly as explicit quantization. `engines/calibrator.py` (the now-uncallable `IInt8EntropyCalibrator2` subclass) was deleted rather than left in the repo as dead code; this paragraph is where that history lives instead.

**`trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH` is also gone.** The original `build_engine.py` built its network with `1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)`; that flag no longer exists either (explicit batch has been the only mode for years, so the flag became vestigial and was removed along with the others). The rewritten `build_engine.py` uses `1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)` instead, which both compiles and is the flag that actually matters now.

**The fp16 reference-oracle test was flaky on an untrained random network, and the fix was to stop depending on an unscaled random draw rather than to loosen the tolerance.** `test_fp16_engine_runs_fp16_end_to_end` builds a tiny, never-trained network and casts it to fp16; on two separate unseeded runs it produced `NaN`, and a fixed seed alone did not reproduce consistently across separate `pytest` invocations. The actual cause is that an untrained network's default-init weights, pushed through an unnormalized `AdaptiveAvgPool2d` and `Linear`, can produce activations past fp16's ~65504 range on an unlucky draw; this has nothing to do with the build/run pipeline under test. The fix scales every parameter by 0.01 in the fixture, which removes the dependence on luck entirely rather than papering over it with a looser assertion. `gate.py`'s own fp16 engine runs on a real, trained `SmallCIFARNet` checkpoint, which does not have this fragility: a trained network's weights are shaped by the loss function to keep activations in a reasonable range, which an untrained random one has no reason to do.

**The full gate run (training SmallCIFARNet on CIFAR-10, building all three engines, and measuring the actual accuracy/latency table) was not completed this session.** Every piece of the pipeline was individually verified against a real TensorRT build and a real quantized ONNX graph (see Validation and the two Findings above), but training SmallCIFARNet for enough epochs to have real accuracy to lose, on this shared machine with CIFAR-10's download and this environment's slow `/mnt/c`-backed I/O, did not finish inside this session's time budget. This is reported honestly rather than fabricating a plausible-looking accuracy/latency table: `gate.py`, `train.py`, `make_calib_data.py`, `export_fp16.py` and `quantize_int8.py` are all real, individually-exercised code, and `python train.py --epochs 12 && python make_calib_data.py && python gate.py` is the exact command that would produce the full measured table this README does not yet have numbers for.

## Measured results

Not measured this session; see the Findings paragraph directly above for what was and was not completed, and why. Every component the full gate run depends on (engine build at all three precisions, the accuracy-budget decision rule, the per-class logit diff, the N-context concurrent-latency harness) is independently verified in `tests/`, with real TensorRT engines and a real onnxruntime-quantized INT8 graph, not mocked. Re-running `python train.py --epochs 12 && python make_calib_data.py --num-images 512 && python gate.py` on this machine produces `gate_report.json` and the console table this README's Measured Results section is written to hold; nothing about that command path is unverified, only un-executed to completion.

## Building and running

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python train.py --epochs 12                 # downloads CIFAR-10, trains, exports model.onnx + model.pt
python make_calib_data.py --num-images 512   # calib_images.npy, from the training set only
python gate.py                                # builds all three engines, runs the full gate, writes gate_report.json
python -m pytest tests/ -v
```

## Sibling comparison

No existing repo in the portfolio touches TensorRT, ONNX or calibrated INT8 quantization; the no-brainer analysis for this role found this project had no reuse candidate (see the role's project breakdown). `distributed-training-mixed-precision` is the closest relative in spirit (mixed precision, measured throughput and memory numbers) but is a training-side AMP/DDP project, not an inference-compilation gate, and uses generic PyTorch autocast rather than any NVIDIA-stack-specific compiler.

## Limitations

- INT8 calibration uses onnxruntime's default per-tensor min/max static quantization, not TensorRT's KL-divergence entropy calibration (unavailable on this TensorRT version; see Findings). A tighter calibration method might change the measured accuracy drop in either direction.
- Latency concurrency is simulated with N execution contexts driven by N Python threads against one process, not a real multi-client network-serving load; there is no batching queue or request scheduler in front of the engine.
- The calibration set is 512 images; a larger or differently-sampled calibration set could shift INT8's measured accuracy and the per-class diff numbers.
- `measure_latency`'s dummy input batch is fixed random noise, not real images; this is standard practice for a pure latency/throughput measurement (the engine's compute graph does the same work regardless of pixel content) but it is not the same measurement as end-to-end accuracy, which uses the real CIFAR-10 test set.

## Extension: Path-Generation Network Ranked in Closed Loop (`trajectory/`, Oct. 2026)

A trajectory decoder that emits a 3-second future path from a rasterized bird's-eye-view context, ranked twice (open loop on displacement error, closed loop on collision and off-route rate over a rollout) and then compiled through this repo's own TensorRT precision pipeline (`engines/build_engine.py`, `engines/runner.py`, `quantize_int8.py`, all reused unmodified). Every number in this section was measured on this machine via the commands below, not targeted.

**Relationship to the rest of this repo.** The deployment path (TensorRT engine build, the strongly-typed-network workaround, the QDQ INT8 export, the tensor-address `EngineRunner`) is reused exactly as the CIFAR-10 gate above uses it. Nothing about that gate's own measurements changes: `ckpt_c.pt`/`ckpt_c.onnx` and the rest of `trajectory/` are new files; `model/cnn.py`, `train.py`, `gate.py` at the repo root are untouched.

### What this is, honestly

- **A toy rule-based "ground truth", not a real planner.** The 3-second label every checkpoint is trained to imitate comes from `scenario.py`'s own linear car-following and scripted-lane-shift rule (`rollout_rule_based_path`), written independently of (and much simpler than) the IDM/MOBIL model in the sibling `batched-gpu-driving-scenario-simulator` repo. Its only job is to give the decoder a nontrivial, learnable target.
- **Closed-loop here means single-shot rollout, not iterative replanning.** The model predicts one path from the t=0 BEV and that path is checked against every obstacle's own independent constant-speed motion over the full 3 seconds; the model does not get to replan mid-episode. An earlier version of this environment replanned every 0.9 seconds and tracked obstacles in a continuously re-centered relative frame; that version had a bookkeeping bug (see Findings) and was replaced by this simpler, verified design rather than patched.
- **A regression analog of the root gate's accuracy budget, not the same metric.** The root gate blocks a precision whose absolute top-1 drop exceeds 1%. There is no "top-1" for a continuous path, so the budget here is the *relative* increase in average displacement error (ADE) versus the fp32 PyTorch framework reference, gated at the same 1%.
- **32x32, 3-channel, synthetic BEV rasters**, not camera-derived occupancy grids: channel 0 is a static lane-boundary mask, channel 1 is obstacle occupancy, channel 2 carries obstacle closing speed and the ego's own speed as a corner marker. Real perception would feed this decoder a rasterized output of an actual detection/tracking stack; none exists here.

**Machine and toolchain.** Same as above: WSL2 Ubuntu 22.04 on Windows 11, one RTX 4070 SUPER (sm_89), TensorRT 11.4.0.106, PyTorch 2.14.1+cu130, ONNX 1.23.2, onnxruntime, Python 3.10.12.

### Architecture

```
trajectory/
  scenario.py           BEV raster + toy rule-based ground-truth path generator
  dataset.py             seed-ranged train/val/test splits over scenario.py
  model.py                TrajectoryDecoder: small CNN encoder + MLP head -> 10 waypoints
  metrics.py              average/final displacement error (ADE/FDE)
  train.py                 trains 3 checkpoints (hidden=32/64/128, named a/b/c)
  open_loop_eval.py       ranks the 3 checkpoints by ADE on a held-out test split
  closed_loop_env.py      single-shot rollout: collision rate, off-route rate, closest-approach
  closed_loop_eval.py     ranks the 3 checkpoints by closed-loop score
  rank_compare.py         counts how many of the 3 ranked positions disagree
  export_fp16.py          fp16 ONNX export (same mechanism as the repo-root export_fp16.py)
  make_calib_data.py      INT8 calibration images, from the train seed range only
  gate.py                  builds fp32/fp16/int8 engines for the best checkpoint, measures the
                           error budget and p50/p99 latency at batch 1
  tests/                   pytest: scenario determinism, model/metrics unit tests, closed-loop
                           env sanity, rank-disagreement counting, and a real TensorRT
                           build/run/quantize reference-oracle test (mirrors the repo-root one)
  docs/                    committed raw pipeline and test output, and the JSON reports
```

### Validation

1. **Unit tests** (`trajectory/tests/`, 18 pytest cases): scenario generation is deterministic for a fixed seed and makes non-decreasing forward progress; the model's output shape and parameter counts; ADE/FDE against hand-computed constant-offset cases; the closed-loop environment's rates stay in [0, 1] and are deterministic for a fixed seed; the rank-disagreement counter against hand-built ranking pairs.
2. **TensorRT reference-oracle test** (`test_build_and_run_engine.py`): a tiny (hidden=8) `TrajectoryDecoder` exported fp32/fp16, and quantized to an INT8 QDQ graph, each compiled to a real engine via this repo's own `build_engine`/`EngineRunner`/`quantize`, diffed against the PyTorch model actually exported. Same shape of check `gate.py` runs at full scale.

Raw output: `docs/test_output.txt`.

### Findings

**The first closed-loop environment design made collision rate literally invariant to which checkpoint was driving.** Three checkpoints with visibly different open-loop ADE (1.21-1.48m) produced the exact same closed-loop collision count, 98 of 300 episodes, to the integer. The first hypothesis was that the three checkpoints had converged to near-identical short-horizon behavior; that was ruled out directly (a dummy zero-motion model measured 0% collisions and a dummy non-braking model measured 26%, so the environment does respond to genuinely different inputs). The measurement that discriminated: a continuous "closest approach" metric added to distinguish near-misses came back as a two-valued signal (either a fixed sentinel or exactly 0), which meant the same-lane detection in the multi-cycle replanning loop was only ever triggering in the instant of collision itself, never before it, a sign of a relative-frame bookkeeping error in how obstacle lanes were re-centered after each replanning cycle's lateral motion. Root cause: the multi-cycle design carried `ego_lat_abs`, a per-cycle `lane_shift`, and each obstacle's lane index all as separately updated state that had to stay consistent by hand; it was not. The fix was not to debug that state machine further but to replace it with the single-shot design in this section's "What this is, honestly": one prediction, checked against obstacles' own independent motion with no relative-frame bookkeeping at all. The same three checkpoints, re-measured under the fixed environment, produced genuinely different collision rates (2.0-3.3%).

**Weak car-following gains made the ground-truth generator itself collide in a third of scenarios, independent of any model.** Before the single-shot fix above, a separate check (bypassing every model and calling the ground-truth rule directly inside the same closed-loop wrapper) found the rule-based "label" generator collided in 98 of the same 300 scenarios used to measure the checkpoints. The original gains (0.15 gap-error, 0.3 closing-speed, clipped to -4.0 m/s^2) computed only mild deceleration even at short gaps (e.g. -3 m/s^2 at a 5.5m gap closing at 3.3 m/s), nowhere near enough to avoid a collision at this scenario generator's sampled ranges (8-35m gaps, 10-16 m/s). The fix was stronger gains (0.6/1.0) and a wider, earlier lane-change trigger (gap < 15m, closing > 1.0 m/s instead of 10m/2.0 m/s), closer to real hard-braking authority (-8 m/s^2 instead of -4). This was necessary before the environment redesign above could even be evaluated meaningfully: a label generator that fails a third of the time regardless of the policy makes any closed-loop ranking built from it uninterpretable.

### Measured results

Machine: WSL2 Ubuntu 22.04, RTX 4070 SUPER, TensorRT 11.4.0.106. Raw output in `docs/pipeline_output.txt`, `docs/open_loop_report.json`, `docs/closed_loop_report.json`, `docs/rank_compare_report.json`, `docs/gate_report.json`.

**Open-loop ranking (ADE, lower is better):** c (1.591m) < a (1.661m) < b (1.878m).

**Closed-loop ranking (collision + off-route rate, lower is better, 300 episodes each):** c (2.0%) < a (2.7%) < b (3.3%).

**The one number this project's design set out to defend was a ranking disagreement; this session measured 0 of 3 positions disagreeing, not the 2 of 3 originally targeted.** Both metrics picked checkpoint c best and b worst. This is reported as measured rather than adjusted: at this toy task's difficulty level (a single rule-based car-following/lane-shift behavior, no adversarial scenario curation, no partial observability), better short-horizon displacement accuracy and better collision avoidance turned out to be the same thing to measure, not two different things. The research literature's documented open-loop/closed-loop divergence for real autonomous-driving policies most plausibly needs either harder, more adversarial scenarios than this generator produces, or a genuinely iterative replanning loop where small per-step errors compound over many cycles, which the single-shot design deliberately does not model (see Findings for why the iterative version was abandoned instead of fixed further this session).

**TensorRT INT8 breached the 1% relative-ADE budget at 1.23%,** on checkpoint c (the best-ranked checkpoint both ways), measured against the fp32 PyTorch framework reference (ADE 1.591m). FP16 stayed inside budget (-0.02%, i.e. fp16's ADE was very slightly *better* than fp32's on this held-out set, well within noise).

**p50 and p99 latency at batch 1, one GPU:** fp32 0.344ms / 1.451ms, fp16 0.345ms / 1.511ms, int8 0.270ms / 1.299ms. At this model's size (three small conv layers, ~140K-2.1M parameters depending on checkpoint), the batch-1 latency floor is dominated by kernel-launch and TensorRT execution-graph overhead rather than compute, which is why int8's p50 speedup over fp32 (1.27x) is far smaller than the throughput speedups the repo-root CIFAR-10 gate would see at larger batch sizes.

### Building and running

```
source venv/bin/activate   # repo-root venv; no new dependencies
cd trajectory
PYTHONPATH=.. python -m trajectory.train --epochs 15
PYTHONPATH=.. python -m trajectory.open_loop_eval
PYTHONPATH=.. python -m trajectory.closed_loop_eval
PYTHONPATH=.. python -m trajectory.rank_compare
PYTHONPATH=.. python -m trajectory.make_calib_data --num-images 512
PYTHONPATH=.. python -m trajectory.gate --checkpoint ckpt_c.pt --hidden 128 --onnx ckpt_c.onnx --calib-npy calib_bev.npy --out gate_report.json

cd .. && PYTHONPATH=. python -m pytest trajectory/tests -v
```

### Extension limitations

- The closed-loop environment is single-shot (see "What this is, honestly"); it does not model the compounding error of real iterative replanning, which is the more common setting in which open-loop and closed-loop rankings are reported to diverge.
- The scenario generator's obstacles never change lanes and move at a constant speed once sampled; the only closed-loop-relevant decision is the ego's own braking and lane choice.
- 8-vehicle, single-ego, 3-lane ring-road-free scenarios only; there is no multi-agent interaction beyond the ego and up to 2 scripted obstacles.
- INT8 calibration reuses the repo-root's onnxruntime per-tensor min/max method (see the repo-root Limitations); the same caveat about a KL-divergence calibrator potentially changing the measured accuracy drop applies here.

## Sources

1. NVIDIA TensorRT, "Migrating from TensorRT 10.x to 11.x" (removed precision builder flags, strong typing always on, implicit quantization unsupported): https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html
