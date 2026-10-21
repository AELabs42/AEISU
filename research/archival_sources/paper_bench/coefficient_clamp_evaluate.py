"""Evaluate the completed no-clamp control with common tiled reconstruction."""
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
from torch.nn import functional as F
sys.path.insert(0, str(HERE))

from common import ROOT, read_rgb, sha256, write_json
from control_ablation_models import ControlAdapter, REFERENCE
from control_ablation_evaluate import archive_prediction, update, validate_metrics
from final_benchmark_data import CONDITIONS, freeze, read_json
from parallel_quality_metrics import ParallelMetrics
from reconstruction import coefficient_clamp, unpack_coefficients

ARCHIVE = Path("F:/Storage/AI Storage/Output Images/QRISP Final Benchmark 2026-10-04")
OUTPUT = ROOT / "output/coefficient_clamp/2026-10-10"
UNCLAMPED = ROOT / (
    "ablations/15_no_coefficient_clamp/Models/"
    "Training Run 2026-10-10 18-04-12 - aeisu_15_no_coefficient_clamp/"
    "Pytorch Models/aeisu_15_no_coefficient_clamp_ep0100.pt")


def load_pair(archive, pair, condition):
    record = read_json(archive / "input_records" / Path(pair["image"]).with_suffix(".json"))
    # The immutable archive was moved from G: to F:. Resolve beneath its new
    # root without editing manifests or any existing saved output images.
    target_path = archive / "targets" / pair["image"]
    source_path = (archive / "inputs" / condition / pair["image"]).with_suffix(".npz")
    if sha256(target_path) != record["target_sha256"]:
        raise ValueError("Target hash mismatch")
    if sha256(source_path) != record["inputs"][condition]["npz_sha256"]:
        raise ValueError("Input hash mismatch")
    with np.load(source_path, allow_pickle=False) as stored:
        source = stored["rgb"]
    target = read_rgb(target_path).astype(np.float32) / 255
    return pair, condition, record, source, target


class ClampAdapter(ControlAdapter):
    @torch.inference_mode()
    def predict_with_stats(self, rgb, precision):
        source = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].cuda().float()
        h, w = source.shape[-2:]
        py, px = 16 + (-h % 32) // 2, 16 + (-w % 32) // 2
        source = F.pad(source, (px, 32 + (-w % 32) - px, py, 32 + (-h % 32) - py),
                       mode="replicate").contiguous(memory_format=torch.channels_last)
        controller = self.model if precision == "fp16" else self.quantized
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
            meta = controller(source)
        raw = unpack_coefficients(meta, 1, 16, 1)
        raw = raw[..., py:py + h, px:px + w].float().reshape(-1)
        applied = coefficient_clamp(raw, self.model.spec["clamp"])
        magnitude = raw.abs()
        sampled = magnitude[torch.linspace(0, raw.numel() - 1, 4096, device="cuda").long()]
        stats = {
            "raw_abs_max": float(magnitude.max()), "raw_rms": float(raw.square().mean().sqrt()),
            "raw_abs_mean": float(magnitude.mean()),
            "raw_abs_p99_sample4096": float(torch.quantile(sampled, .99)),
            "raw_fraction_above2": float((magnitude > 2).float().mean()),
            "raw_fraction_above12": float((magnitude > 12).float().mean()),
            "applied_abs_max": float(applied.abs().max()),
            "applied_rms": float(applied.square().mean().sqrt()),
        }
        # Both checkpoints use the same FP32 tiled synthesis. The existing
        # fused CUDA reconstruction always clamps, so must not be used here.
        prediction = self.model.differentiable_predict_rgb(source, meta)
        prediction = prediction[:, :, py * 2:(py + h) * 2, px * 2:(px + w) * 2]
        stats["rgb_fraction_outside01_before_clip"] = float(
            ((prediction < 0) | (prediction > 1)).float().mean())
        prediction = prediction.clamp(0, 1).float()[0].permute(1, 2, 0).cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite reconstruction")
        return prediction, stats

    @torch.inference_mode()
    def check(self):
        rng = np.random.default_rng(20261010)
        image = rng.random((32, 40, 3), dtype=np.float32)
        outputs = {p: self.predict_with_stats(image, p)[0] for p in ("fp16", "fp8_stress")}
        if any(v.shape != (64, 80, 3) or not np.isfinite(v).all() for v in outputs.values()):
            raise ValueError("Invalid prediction shape")
        result = {"finite_outputs": True, "output_shape": [64, 80, 3],
                  "clamp": self.model.spec["clamp"], "backend": "common tiled FP32 reconstruction"}
        if self.model.spec["clamp"] == "soft":
            for precision, value in outputs.items():
                error = float(np.abs(value - super().predict(image, precision)).max())
                if error > 5e-5:
                    raise ValueError(f"Fused/tiled baseline mismatch: {error}")
                result[precision + "_fused_vs_tiled_max_abs"] = error
        # Explicitly exercise coefficients outside the transfer's identity
        # interval; the checkpoint's resolved mode must remain identity.
        probe = torch.tensor([-100., -3., 0., 3., 100.], device="cuda")
        expected = probe if self.name == "unclamped" else coefficient_clamp(probe, "soft")
        torch.testing.assert_close(coefficient_clamp(probe, self.model.spec["clamp"]), expected)
        result["transfer_identity_or_soft_test"] = True
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--checkpoint-manifest", type=Path)
    args = parser.parse_args()
    if torch.version.cuda != "13.0" or args.batch < 1:
        raise ValueError("Requires Torch CUDA 13.0 and a positive batch")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.archive / "selection.json"
    pairs = read_json(selection_path)["pairs"]
    if len(pairs) != 600:
        raise ValueError("Expected frozen 600-pair selection")
    specified = read_json(args.checkpoint_manifest) if args.checkpoint_manifest else {}
    models = {
        "clamped": ClampAdapter("clamped", specified.get("clamped", REFERENCE), gradient_phase="fixed", expected_settings={}),
        "unclamped": ClampAdapter("unclamped", specified.get("unclamped", UNCLAMPED),
                                  gradient_phase="fixed" if specified else "cycle", expected_settings={"clamp": "none"}),
    }
    paths = sorted((ROOT / "data/train/Diverse Input").glob("*.png"))
    if len(paths) != 700:
        raise ValueError("Expected 700 training pairs")
    calibration = [paths[int((i + .5) * len(paths) / 24)] for i in range(24)]
    checks = {}
    for name, model in models.items():
        print(f"Calibrating and checking {name}", flush=True)
        model.calibrate(calibration)
        checks[name] = model.check()
    metric = ParallelMetrics()
    rng = np.random.default_rng(20261010)
    checks["metrics"] = validate_metrics(metric, [rng.random((64, 80, 3), dtype=np.float32) for _ in range(2)])
    slots = [(m, "fp16", c) for m in models for c in CONDITIONS]
    slots += [(m, "fp8_stress", "ssaa_area") for m in models]
    protocol = {
        "selection": str(selection_path), "selection_sha256": sha256(selection_path), "pairs": 600,
        "conditions": CONDITIONS, "slots": slots, "models": {n: m.metadata for n, m in models.items()},
        "evaluation": "Common FP16 controller / FP32 analysis, constants and tiled reconstruction; align32 + halo16; clipped float32",
        "quantization": "Area-only E4M3 rounding simulation; per-output-channel weights, per-layer per-tensor calibrated inputs; FP32 arithmetic, no bias correction",
        "calibration": [{"path": str(p), "sha256": sha256(p)} for p in calibration],
        "calibration_crops": "Same 24 fixed centered 128x128 LR training crops for both checkpoints",
        "coefficient_statistics": "All unpadded LR sites; exact maxima, means and threshold fractions; deterministic 4096-value p99 sample per image",
        "training_caveat": "Both 100 epochs, 87500 updates, seed1234, area700 pairs, batch32, HR512, D4, MSE, cosine2e-3 to1e-6. "
                           + "; ".join(f"{n}: {m.metadata['gradient_phase']} offsets, {m.metadata['training_backend']}"
                                       for n, m in models.items()) + ". One training seed.",
        "metrics": metric.perceptual.protocol,
        "archive_relocation": "Resolve records beneath the moved archive root; validate original input/target hashes without editing old files",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "sources": {p.name: sha256(p) for p in (Path(__file__), HERE / "control_ablation_models.py",
                                                  ROOT / "ablations/_shared/reconstruction.py")},
    }
    freeze(args.output / "protocol.json", json.loads(json.dumps(protocol)))
    write_json(args.output / "preflight.json", checks)
    # Confirm archive remapping and content before committing any benchmark.
    load_pair(args.archive, pairs[0], "ssaa_area")
    print("Checkpoint, folding, transfer, reconstruction, metrics and input preflights passed.", flush=True)
    if args.preflight_only:
        return
    results = args.output / "per_image.jsonl"
    rows = [json.loads(x) for x in results.read_text().splitlines() if x.strip()] if results.exists() else []
    done = {(r["method"], r["precision"], r["condition"], r["image"]) for r in rows}
    expected = {(*slot, p["image"]) for slot in slots for p in pairs}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("Invalid existing scores")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as io, results.open("a", encoding="utf-8", buffering=1) as stream:
        for name, precision, condition in slots:
            remaining = [p for p in pairs if (name, precision, condition, p["image"]) not in done]
            for first in range(0, len(remaining), args.batch):
                loaded = list(io.map(lambda p: load_pair(args.archive, p, condition), remaining[first:first + args.batch]))
                predictions, stats = zip(*(models[name].predict_with_stats(item[3], precision) for item in loaded))
                scores = metric.score_batch(list(predictions), [item[4] for item in loaded])
                futures = [io.submit(archive_prediction, args.output, name, precision, item, pred, score)
                           for item, pred, score in zip(loaded, predictions, scores)]
                for future, statistic in zip(futures, stats):
                    row = future.result() | {"coefficients": statistic}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    done.add((name, precision, condition, row["image"]))
                update(args.output, phase="evaluating", completed=len(done), total=len(expected), model=name,
                       precision=precision, condition=condition, elapsed_seconds=time.perf_counter() - started)
                if len(done) % 40 == 0 or len(done) == len(expected):
                    print(f"{len(done)}/{len(expected)} | {name} {precision} {condition} | {time.perf_counter() - started:.1f}s", flush=True)
    if done != expected:
        raise ValueError("Incomplete evaluation")
    update(args.output, phase="evaluation_complete", completed=len(done), total=len(expected))
    print("All 7200 reconstructions, scores and coefficient summaries saved.", flush=True)


if __name__ == "__main__":
    main()
