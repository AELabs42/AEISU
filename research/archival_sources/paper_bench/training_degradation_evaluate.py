"""Compare the completed Area/Bicubic main checkpoints on frozen QRISP inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
import numpy as np
import torch
sys.path.insert(0, str(HERE))

from common import ROOT, RUN, sha256, write_json
from control_ablation_models import ControlAdapter
from control_ablation_evaluate import archive_prediction, prefetched, update, validate_metrics
from final_benchmark_data import CONDITIONS, DEFAULT_OUTPUT, freeze, read_json
from parallel_quality_metrics import ParallelMetrics

OUTPUT = Path("G:/Output Images/AEISU Training Degradation 2026-10-06")


def model_paths():
    candidates = list((ROOT / "Models/AEISU Bicubic 8K").glob("Training Run */Pytorch Models/*_ep1000.pt"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one completed Bicubic run, found {len(candidates)}")
    return {"area_trained": RUN / "Pytorch Models/aif_reparam_aggressive64_parallel32x4_ep1000.pt",
            "bicubic_trained": candidates[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--checkpoint-manifest", type=Path)
    args = parser.parse_args()
    if torch.version.cuda != "13.0" or args.batch < 1:
        raise ValueError("Requires Torch CUDA 13.0 and positive batch size")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.archive / "selection.json"
    pairs = read_json(selection_path)["pairs"]
    if len(pairs) != 600:
        raise ValueError("Expected frozen 600-pair QRISP selection")
    paths = read_json(args.checkpoint_manifest) if args.checkpoint_manifest else model_paths()
    models = {name: ControlAdapter(name, path, 1000,
                                  "fixed" if args.checkpoint_manifest or name == "area_trained" else "cycle")
              for name, path in paths.items()}
    metric = ParallelMetrics()
    checks = {name: model.validate(("fp16",)) for name, model in models.items()}
    rng = np.random.default_rng(20261006)
    checks["metrics"] = validate_metrics(metric, [rng.random((64, 80, 3), dtype=np.float32) for _ in range(2)])
    cfgs = [m.metadata["config"] for m in models.values()]
    shared_keys = ("epochs", "batch", "patch", "source_pad", "seed", "lr", "min_lr", "optimizer", "weight_decay",
                   "adam_beta1", "adam_beta2", "adam_eps", "grad_clip", "basis_grad_stride", "amp_dtype")
    shared = {key: cfgs[0][key] for key in shared_keys}
    if any(cfgs[1][key] != value for key, value in shared.items()):
        raise ValueError("Unexpected budget/optimizer mismatch")
    slots = [(name, "fp16", condition) for name in models for condition in CONDITIONS]
    protocol = {"selection": str(selection_path), "selection_sha256": sha256(selection_path), "pairs": len(pairs),
                "conditions": CONDITIONS, "slots": slots, "models": {n: m.metadata for n, m in models.items()},
                "shared_training_config": shared,
                "evaluation": "Common CPU-FP32 folding, FP16 controller, FP32 analysis/constants/fused RGB reconstruction; align32+halo16",
                "metrics": metric.perceptual.protocol | {"distortion": "Encoded BT.601 Y and RGB, FP64; full-frame and shave2; Gaussian11 sigma1.5 valid SSIM"},
                "training_caveat": "Same 700 target identities, budget and seed. Gradient policies: "
                                   + ", ".join(f"{n}={m.metadata['gradient_phase']}" for n, m in models.items())
                                   + ". Area input preparation and direct 8K MATLAB-style Bicubic differ; QRISP evaluation Bicubic is Pillow antialiased Bicubic.",
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "source_sha256": {p.name: sha256(p) for p in (Path(__file__), HERE / "control_ablation_models.py",
                                    HERE / "parallel_quality_metrics.py", HERE / "final_benchmark_metrics.py")}}
    freeze(args.output / "protocol.json", json.loads(json.dumps(protocol)))
    write_json(args.output / "preflight.json", checks)
    print("Both 1000-epoch checkpoint, folding, reconstruction and metric checks passed.", flush=True)
    if args.preflight_only:
        return
    total = len(pairs) * len(slots)
    results = args.output / "per_image.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()] if results.exists() else []
    done = {(r["method"], r["precision"], r["condition"], r["image"]) for r in rows}
    expected = {(*slot, p["image"]) for slot in slots for p in pairs}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("Invalid existing result set")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as io, results.open("a", encoding="utf-8", buffering=1) as stream:
        for name, model in models.items():
            for condition in CONDITIONS:
                remaining = [p for p in pairs if (name, "fp16", condition, p["image"]) not in done]
                batch = []

                def process(items):
                    predictions = [model.predict(item[3]) for item in items]
                    scores = metric.score_batch(predictions, [item[4] for item in items])
                    futures = [io.submit(archive_prediction, args.output, name, "fp16", item, prediction, score)
                               for item, prediction, score in zip(items, predictions, scores)]
                    for future in futures:
                        row = future.result()
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        done.add((name, "fp16", condition, row["image"]))
                    update(args.output, phase="evaluating", completed=len(done), total=total, model=name,
                           condition=condition, elapsed_seconds=time.perf_counter()-started)
                    if len(done) % 40 == 0:
                        print(f"{len(done)}/{total} | {name} {condition} | {time.perf_counter()-started:.1f}s", flush=True)

                for item in prefetched(io, args.archive, remaining, condition, args.batch * 2):
                    batch.append(item)
                    if len(batch) == args.batch:
                        process(batch)
                        batch = []
                if batch:
                    process(batch)
    if done != expected:
        raise ValueError("Incomplete evaluation")
    update(args.output, phase="evaluation_complete", completed=total, total=total)
    print("All 6000 reconstructions and metrics saved.", flush=True)


if __name__ == "__main__":
    main()
