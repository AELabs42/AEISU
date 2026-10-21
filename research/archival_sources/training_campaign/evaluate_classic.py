"""Evaluate the new DIV2K checkpoint and released comparators on 219 immutable bicubic pairs."""
from __future__ import annotations

import csv
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data/benchmark_assets/classic_comparators"
os.environ["AEISU_BENCHMARK_ROOT"] = str(BENCH)
PAPER_BENCH = ROOT / "tools/paper_bench"
sys.path[:0] = [str(PAPER_BENCH), str(BENCH)]
# The neighboring paper-benchmark profile.py must not shadow stdlib profile.
sys.path.remove(str(PAPER_BENCH))
import numpy as np
import torch
sys.path.insert(0, str(PAPER_BENCH))
from common import read_rgb, sha256, write_json
from control_ablation_models import ControlAdapter
from final_benchmark_models import Adapter
from final_benchmark_metrics import PerceptualMetric, distortion
from final_benchmark_data import freeze, save_rgb
from luma_metrics import y_channel
from train_all import completed_checkpoint

SETS = {"Set5": 5, "Set14": 14, "BSD100": 100, "Urban100": 100}
METHODS = ("aeisu_div2k_fp16", "fsrcnn_x2", "bilinear", "fsr1_easu", "repnet_x2", "rt4ksr_x2",
           "quicksrnet_small_x2", "quicksrnet_medium_x2", "span_x2", "swinir_light_x2")


def pairs():
    manifest = BENCH / "manifests/dataset_manifest_matlab_219.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 219 or {name: sum(r["dataset"] == name for r in records) for name in SETS} != SETS:
        raise ValueError("Expected unchanged Set5/Set14/BSD100/Urban100 219-image protocol")
    for record in records:
        # Older manifests may call this set B100 rather than BSD100.
        for key in ("lr", "hr"):
            path = Path(record[key + "_path"])
            if sha256(path) != record[key + "_sha256"]:
                raise ValueError(f"Frozen benchmark image changed: {path}")
    return records


class FSRCNNAdapter:
    def __init__(self):
        from models_arch.fsrcnn import FSRCNN
        path = BENCH / "checkpoints/FSRCNN/fsrcnn_x2.pth"
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model = FSRCNN(2).eval().requires_grad_(False)
        self.model.load_state_dict(state.get("state_dict", state), strict=True)
        self.model.cuda().float()
        self.metadata = {"checkpoint": str(path), "checkpoint_sha256": sha256(path), "precision": "FP32",
                         "input": "BT.601 limited-range Y normalized by 255", "chroma": "PyTorch FP32 bicubic RGB chroma carrier",
                         "reference": "https://github.com/yjn870/FSRCNN-pytorch/blob/master/test.py",
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "scope": "Recomputed common RGB-output metrics; not a reproduction of published rounded Y-only scores"}

    @torch.inference_mode()
    def infer(self, low):
        source = torch.from_numpy(np.ascontiguousarray(low)).permute(2, 0, 1)[None].cuda()
        carrier = torch.nn.functional.interpolate(source, scale_factor=2, mode="bicubic", align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
        y = torch.from_numpy(y_channel(low).astype(np.float32))[None, None].cuda()
        predicted = self.model(y).clamp(0, 1)[0, 0].cpu().numpy()
        # Preserve the carrier's chroma while replacing its limited-range Y.
        rgb = carrier + ((predicted - y_channel(carrier)) * (255 / 219))[..., None]
        return rgb.clip(0, 1).astype(np.float32)


def evaluate(campaign, output):
    if torch.version.cuda != "13.0":
        raise RuntimeError("Torch CUDA 13.0 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    records = pairs()
    checkpoint = completed_checkpoint(campaign, "div2k", 1000)
    output.mkdir(parents=True, exist_ok=True)
    metric = PerceptualMetric()
    protocol = {"training_checkpoint": checkpoint, "training_checkpoint_sha256": sha256(Path(checkpoint)),
                "training_manifest_sha256": sha256(campaign / "div2k/pairs.json"), "sets": SETS, "images": records,
                "methods": METHODS, "metrics": metric.protocol,
                "distortion": "BT.601 encoded Y and RGB PSNR/SSIM, float64, shave2 primary; no output resizing",
                "input": "Existing hashed MATLAB-style antialiased bicubic X2 LR, not Area inputs",
                "selection": "All 219 test images; final epoch selected without test feedback",
                "scope": "Quality only. Released comparators retain their original training data and objectives. No latency claims.",
                "source_sha256": sha256(Path(__file__))}
    freeze(output / "protocol.json", json.loads(json.dumps(protocol)))
    results_path = output / "per_image.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()] if results_path.exists() else []
    expected = {(m, r["dataset"], r["hr_filename"]) for m in METHODS for r in records}
    done = {(r["method"], r["dataset"], r["image"]) for r in rows}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("Invalid existing score identities")
    metadata = {}
    with results_path.open("a", encoding="utf-8", buffering=1) as stream:
        for name in METHODS:
            if all((name, r["dataset"], r["hr_filename"]) in done for r in records):
                continue
            if name == "aeisu_div2k_fp16":
                model = ControlAdapter(name, checkpoint, 1000, "fixed", expected_settings={}, expected_updates=0)
                predict = model.predict
            elif name == "fsrcnn_x2":
                model = FSRCNNAdapter()
                predict = model.infer
            else:
                model = Adapter(name, output)
                predict = model.infer
            metadata[name] = model.metadata
            freeze(output / "models" / (name + ".json"), model.metadata)
            for record in records:
                key = name, record["dataset"], record["hr_filename"]
                if key in done:
                    continue
                low = read_rgb(Path(record["lr_path"])).astype(np.float32) / 255
                high = read_rgb(Path(record["hr_path"])).astype(np.float32) / 255
                h, w = low.shape[:2]
                # Standard scale-modcrop: no resampling, top-left retained.
                if high.shape[0] not in (h * 2, h * 2 + 1) or high.shape[1] not in (w * 2, w * 2 + 1):
                    raise ValueError("Unexpected benchmark pair alignment")
                high = high[:h * 2, :w * 2]
                prediction = predict(low).astype(np.float32)
                values = distortion(prediction, high) | {"lpips_alex_v01": metric.score(prediction, high)}
                saved = save_rgb(prediction, output / "outputs" / name / record["dataset"] / record["hr_filename"], verify=True)
                row = {"method": name, "dataset": record["dataset"], "image": record["hr_filename"],
                       "metrics": values, "outputs": saved, "input_sha256": record["lr_sha256"], "target_sha256": record["hr_sha256"]}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                done.add(key)
                write_json(output / "status.json", {"phase": "evaluating", "completed": len(done), "total": len(expected), "method": name})
                print(f"{len(done)}/{len(expected)} {name} {record['dataset']} {record['hr_filename']}", flush=True)
            del model, predict
            gc.collect()
            torch.cuda.empty_cache()
    if done != expected:
        raise ValueError("Incomplete classic-set evaluation")
    for row in rows:
        for kind in ("npz", "png"):
            if sha256(Path(row["outputs"][kind])) != row["outputs"][kind + "_sha256"]:
                raise ValueError("Output archive verification failed")
    means = {dataset: {name: {k: float(np.mean([r["metrics"][k] for r in rows if r["dataset"] == dataset and r["method"] == name]))
                              for k in rows[0]["metrics"]} for name in METHODS} for dataset in SETS}
    write_json(output / "summary.json", {"complete": True, "images": 219, "reconstructions": len(rows), "means": means})
    lines = ["# DIV2K-trained AEISU: classic bicubic X2 evaluation", "", protocol["scope"], "",
             "800 official DIV2K training pairs; 1,000 epochs, fixed stride-16 offsets, D4, MSE, cosine 2e-3 to 1e-6. "
             "All 219 test images are evaluation-only. FP16-controller/FP32-reconstruction AEISU; released neural adapters "
             "use their recorded precision. No mixed-runtime result is inferred from this checkpoint test.", "",
             "Primary Y metrics shave two HR pixels; LPIPS-Alex v0.1 uses cropped RGB. B100 is labeled BSD100 in the archive."]
    for dataset in SETS:
        lines += ["", "## " + dataset, "", "| Method | Y PSNR | Y SSIM | LPIPS |", "|---|---:|---:|---:|"]
        for name in METHODS:
            v = means[dataset][name]
            lines.append(f"| {name} | {v['y_psnr_shave2_db']:.5f} | {v['y_ssim_shave2']:.7f} | {v['lpips_alex_v01']:.7f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "verification.json", {"passed": True, "coverage": len(expected), "all_output_hashes_verified": True})
    write_json(output / "status.json", {"phase": "complete", "completed": len(done), "total": len(expected)})
