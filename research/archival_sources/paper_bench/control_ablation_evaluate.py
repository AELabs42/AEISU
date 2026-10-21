"""Evaluate the completed core controls on the immutable diverse QRISP inputs."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

# Avoid the neighboring profile.py shadowing Python's standard-library module.
HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
import numpy as np
import torch
sys.path.insert(0, str(HERE))

from common import ROOT, read_rgb, sha256, write_json
from control_ablation_models import ControlAdapter, VARIANTS
from final_benchmark_data import CONDITIONS, DEFAULT_OUTPUT, freeze, input_record, load_rgb, read_json, save_rgb
from parallel_quality_metrics import ParallelMetrics
from final_benchmark_metrics import distortion

DESTINATION = Path("G:/Output Images/AEISU Core Ablations 2026-10-05/evaluation_v2")


def update(output, **fields):
    write_json(output / "status.json", {"pid": os.getpid(), "updated_unix": time.time(), **fields})


def load_job(archive, pair, condition):
    record = input_record(archive, pair)
    source_record = record["inputs"][condition]
    source_path = (archive / "inputs" / condition / pair["image"]).with_suffix(".npz")
    target_path = archive / "targets" / pair["image"]
    if sha256(source_path) != source_record["npz_sha256"] or sha256(target_path) != record["target_sha256"]:
        raise ValueError("Frozen evaluation input changed")
    source = load_rgb(source_record | {"npz": str(source_path)})
    target = read_rgb(target_path).astype(np.float32) / 255
    return pair, condition, record, source, target


def archive_prediction(output, name, precision, loaded, prediction, metrics):
    pair, condition, record, _, _ = loaded
    base = (output / "outputs" / name / precision / condition / pair["image"]).with_suffix("")
    row = {key: pair[key] for key in ("image", "scene", "segment", "frame", "group")}
    return row | {"method": name, "precision": precision, "condition": condition, "metrics": metrics,
                  "target_sha256": record["target_sha256"], "input_sha256": record["inputs"][condition]["npz_sha256"],
                  "outputs": save_rgb(prediction, base)}


def prefetched(pool, archive, pairs, condition, depth):
    iterator = iter(pairs)
    pending = deque()
    for _ in range(depth):
        pair = next(iterator, None)
        if pair is not None:
            pending.append(pool.submit(load_job, archive, pair, condition))
    while pending:
        item = pending.popleft().result()
        pair = next(iterator, None)
        if pair is not None:
            pending.append(pool.submit(load_job, archive, pair, condition))
        yield item


def validate_metrics(metric, targets):
    predictions = [target * .97 for target in targets]
    expected = [distortion(p, t) | {"lpips_alex_v01": metric.perceptual.score(p, t)} for p, t in zip(predictions, targets)]
    actual = metric.score_batch(predictions, targets)
    differences = {}
    for left, right in zip(expected, actual):
        for key, value in left.items():
            error = abs(value - right[key])
            differences[key] = max(differences.get(key, 0.), error)
            tolerance = 1e-6 if key == "lpips_alex_v01" else 2e-10
            if error > tolerance:
                raise ValueError(f"Metric parity failed: {key} {error}")
    return {"passed": True, "maximum_absolute_differences": differences, "lpips_tolerance": 1e-6,
            "distortion_tolerance": 2e-10, "note": "LPIPS batch-dependent FP32 convolution rounding; metric definition unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--checkpoint-manifest", type=Path, help="Explicit checkpoint identities; never select by newest folder")
    args = parser.parse_args()
    if torch.version.cuda != "13.0":
        raise RuntimeError("Torch CUDA 13.0 is required")
    if args.batch < 1:
        parser.error("Batch must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.archive / "selection.json")
    if len(selection["pairs"]) != 600:
        raise ValueError("Expected the unchanged 600-pair selection")
    paths = sorted((ROOT / "data/train/Diverse Input").glob("*.png"))
    calibration = [paths[int((i + .5) * len(paths) / 24)] for i in range(24)]
    specified = read_json(args.checkpoint_manifest) if args.checkpoint_manifest else {}
    settings = {"reference": {}, "fixed_inputs": {"analysis": "fixed"}, "fixed_base": {"learned_base": False},
                "dense_core": {"backbone": "dense64x2"}}
    models = {name: ControlAdapter(name, specified.get(name), expected_settings=settings[name]) for name in VARIANTS}
    metric = ParallelMetrics()
    protocol = {"selection": str(args.archive / "selection.json"), "selection_sha256": sha256(args.archive / "selection.json"),
                "pairs": 600, "conditions": CONDITIONS, "models": {name: model.metadata for name, model in models.items()},
                "primary": "FP16 controller convolutions; FP32 analysis, filter constants and fused reconstruction; clipped float32 outputs",
                "padding": "Established Eval.py symmetric align32, replicate halo16, crop back to exact output size; no input rounding",
                "quantization": "Area-only uniform controller FP8 E4M3 rounding simulation: per-output-channel weights, per-layer per-tensor activation scales from calibration maxima; FP32 convolution accumulation, biases, analysis and reconstruction; no bias correction; not a native latency or mixed-runtime test",
                "calibration": [{"path": str(p), "sha256": sha256(p)} for p in calibration],
                "calibration_crops": "24 fixed centered 128x128 LR training crops; shared identities and budget across models",
                "metrics": metric.perceptual.protocol | {"distortion": "FP64 RGB and BT.601 encoded Y, full frame and shave2; Gaussian 11x11 sigma1.5 SSIM valid windows"},
                "training_caveat": "Gradient policies: " + ", ".join(f"{n}={m.metadata['gradient_phase']}" for n, m in models.items())
                                   + ". One training seed; scene uncertainty is not seed uncertainty.",
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "sources": {p.name: sha256(p) for p in (Path(__file__), HERE / "control_ablation_models.py", HERE / "parallel_quality_metrics.py")}}
    checks = {}
    for name, model in models.items():
        print(f"Calibrating and validating {name}", flush=True)
        model.calibrate(calibration)
        checks[name] = model.validate()
    protocol["models"] = {name: model.metadata for name, model in models.items()}
    freeze(args.output / "protocol.json", json.loads(json.dumps(protocol)))
    rng = np.random.default_rng(20261005)
    targets = [rng.random((64, 80, 3), dtype=np.float32) for _ in range(2)]
    checks["metrics"] = validate_metrics(metric, targets)
    write_json(args.output / "preflight.json", checks)
    print("All checkpoint, folding, reconstruction and metric preflights passed.", flush=True)
    if args.preflight_only:
        return
    total = 600 * len(VARIANTS) * (len(CONDITIONS) + 1)
    committed = []
    result_path = args.output / "per_image.jsonl"
    if result_path.exists():
        committed = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {(r["method"], r["precision"], r["condition"], r["image"]) for r in committed}
    if len(done) != len(committed):
        raise ValueError("Duplicate committed results")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as io, result_path.open("a", encoding="utf-8", buffering=1) as stream:
        for precision, conditions in (("fp16", CONDITIONS), ("fp8_stress", ("ssaa_area",))):
            for name, model in models.items():
                for condition in conditions:
                    remaining = [p for p in selection["pairs"] if (name, precision, condition, p["image"]) not in done]
                    loaded = prefetched(io, args.archive, remaining, condition, args.batch * 2)
                    batch = []

                    def process(items):
                        predictions = [model.predict(item[3], precision) for item in items]
                        scores = metric.score_batch(predictions, [item[4] for item in items])
                        futures = [io.submit(archive_prediction, args.output, name, precision, item, prediction, score)
                                   for item, prediction, score in zip(items, predictions, scores)]
                        for future in futures:
                            row = future.result()
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            done.add((name, precision, condition, row["image"]))
                        update(args.output, phase="evaluating", completed=len(done), total=total, model=name,
                               precision=precision, condition=condition, elapsed_seconds=time.perf_counter() - started)
                        if len(done) % 40 == 0 or len(done) == total:
                            print(f"{len(done)}/{total} | {name} {precision} {condition} | {time.perf_counter() - started:.1f}s", flush=True)

                    for item in loaded:
                        batch.append(item)
                        if len(batch) == args.batch:
                            process(batch)
                            batch = []
                    if batch:
                        process(batch)
    if len(done) != total:
        raise ValueError("Incomplete evaluation")
    update(args.output, phase="evaluation_complete", completed=len(done), total=total)
    print("Evaluation complete; every reconstruction and score is saved.", flush=True)


if __name__ == "__main__":
    main()
