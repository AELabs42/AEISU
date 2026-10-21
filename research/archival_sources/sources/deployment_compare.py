"""Matched resident-image timing of Tensor Core and conventional CUDA deployment."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

import cv2
import torch

from models import ROOT, CheckpointRuntime, load_shader_extension
from artifacts import SavedShaderRuntime

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import latency, instruction_audit
from common import CHECKPOINT, clear_native_scratch, native, provenance, read_rgb, sha256, write_json
from timing import gpu_status, parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/side_runtime/2026-10-04_deployment_compare")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--input", type=Path, default=ROOT / "data/Validation/1080p/Area/scene0045.png")
    args = parser.parse_args()
    folder = args.output.resolve()
    if (folder / "trials.json").exists():
        raise FileExistsError("Preserve completed measurements; choose a new output directory")
    folder.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    extension = load_shader_extension(fused=True)
    audit = instruction_audit(extension, folder)
    artifact = ROOT / "output/side_runtime/2026-10-04_precision_sweep/models/shader_profile_selected_mixed.pt"
    models = {"shader_int8_fp16": SavedShaderRuntime(artifact, extension),
              "shader_fp16": CheckpointRuntime(CHECKPOINT, extension=extension, fusion=True,
                                               shared_phases=True, head_block=256)}
    path = args.input.resolve()
    image = read_rgb(path).astype("float32") / 255
    protocol = {"model": provenance(), "shader_artifact": str(artifact), "shader_artifact_sha256": sha256(artifact),
                "input": str(path), "input_sha256": sha256(path), "resize": "OpenCV INTER_LINEAR",
                "hardware_start": gpu_status(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "batch": 1, "warmup": args.warmup, "iterations": args.iterations, "trials": args.trials,
                "execution": "CUDA graph replay",
                "scope": "Complete resident input to clipped RGB; analysis/controller/reconstruction included",
                "layout": "Native RGBA32F and shader contiguous NCHW FP32; same RGB values and dimensions",
                "aggregation": "Median of trial medians; median of within-trial p95 values",
                "order": "Per resolution/trial: native fixed FP8/FP16/mixed precision order, then shader INT8/FP16, shader FP16",
                "instruction_audit": audit, "source_sha256": sha256(Path(__file__))}
    write_json(folder / "protocol.json", protocol)
    rows = []
    for width, height in ((960, 540), (1920, 1080)):
        rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        source = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda().contiguous()
        for trial in range(args.trials):
            torch.cuda.synchronize()
            directory = folder / "native" / f"{width}x{height}_trial{trial}"
            _, log = native(rgb, directory, warmup=args.warmup, iterations=args.iterations, output=False)
            clear_native_scratch(directory)
            native_scores = parse(log)
            for precision in ("mixed", "fp16"):
                rows.append({"input": [width, height], "trial": trial, "route": f"tensor_{precision}",
                             "result": native_scores[precision]})
            for name, model in models.items():
                result = latency(model, source, args.warmup, args.iterations, 1, graph=True)
                rows.append({"input": [width, height], "trial": trial, "route": name, "result": result})
            write_json(folder / "trials.json", rows)
            print(f"Timing {width}x{height} trial {trial + 1}/{args.trials} completed", flush=True)
    summary = []
    for width, height in ((960, 540), (1920, 1080)):
        for route in ("tensor_mixed", "tensor_fp16", *models):
            measured = [r["result"] for r in rows if r["input"] == [width, height] and r["route"] == route]
            summary.append({"input": [width, height], "route": route,
                            "median_ms": statistics.median(r["median_ms"] for r in measured),
                            "p95_ms": statistics.median(r["p95_ms"] for r in measured),
                            "median_range_ms": [min(r["median_ms"] for r in measured), max(r["median_ms"] for r in measured)]})
    write_json(folder / "summary.json", summary)
    write_json(folder / "hardware_end.json", gpu_status())
    print(summary, flush=True)


if __name__ == "__main__":
    main()
