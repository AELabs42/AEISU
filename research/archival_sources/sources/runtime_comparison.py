"""Isolated-process, resident-image PyTorch eager / CUDA-graph timing campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

# Prevent the adjacent diagnostic profile.py from shadowing stdlib profile.
HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != HERE]
import cProfile
import torch
import cv2
import numpy as np
sys.path.insert(0, str(HERE))

from common import ROOT, read_rgb, sha256, write_json
from timing import gpu_status
from runtime_torch_models import MODELS, difference, load


def measure(run, iterations):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start, end in events:
        start.record()
        value = run()
        end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - started) * 1000 / iterations
    samples = [start.elapsed_time(end) for start, end in events]
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95)),
            "synchronized_loop_ms_per_frame": wall, "samples_ms": samples,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


@torch.inference_mode()
def worker(args):
    folder = args.output / "pytorch" / f"{args.model}_{args.height}"
    if (folder / "summary.json").exists():
        print(f"Preserving completed {args.model} {args.height}", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.version.cuda != "13.0":
        raise RuntimeError("This experiment requires PyTorch CUDA 13.0")
    source_path = ROOT / "data/Benchmark Data/Urban100/HR/img001.png"
    image = read_rgb(source_path).astype(np.float32) / 255
    width = args.height * 16 // 9
    rgb = cv2.resize(image, (width, args.height), interpolation=cv2.INTER_LINEAR)
    dtype = torch.float32 if args.model.endswith("fp32_synthesis") else torch.float16
    source = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda().to(dtype).contiguous(memory_format=torch.channels_last)
    fn, metadata, preflight = load(args.model, folder)
    protocol = {"model": args.model, "metadata": metadata, "preflight": preflight,
                "input": str(source_path), "input_sha256": sha256(source_path), "input_shape_wh": [width, args.height],
                "output_shape_wh": [2 * width, 2 * args.height], "input_preparation": "RGB/255; OpenCV INTER_LINEAR",
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "device": torch.cuda.get_device_name(), "hardware_start": gpu_status(), "batch": 1,
                "precision": "FP16" if dtype == torch.float16 else "FP16 controller / FP32 analysis and synthesis",
                "channels_last": True, "tf32": False, "cudnn_autotune": True, "torch_compile": False,
                "custom_cuda_extensions": False, "image_tiling": False,
                "warmup": args.warmup, "iterations": args.iterations, "trials": args.trials,
                "timing": "Resident input to complete clipped RGB; all dynamic padding/analysis/reconstruction included; "
                          "no disk I/O or CPU/GPU image transfers in interval",
                "execution": ["eager", "cuda_graph"], "order": "Alternates eager/graph per trial",
                "aggregation": "Median of trial medians and trial p95 values; raw samples and synchronized-loop throughput retained",
                "script_sha256": sha256(Path(__file__)), "adapter_sha256": sha256(HERE / "runtime_torch_models.py")}
    write_json(folder / "protocol.json", protocol)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            expected = fn(source)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if expected.shape != (1, 3, args.height * 2, width * 2) or not torch.isfinite(expected).all():
        raise RuntimeError("Full-frame geometry or finite-value preflight failed")
    expected = expected.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn(source)
    graph.replay()
    torch.cuda.synchronize()
    graph_check = difference(expected, captured)
    if graph_check["max_abs"] > 0.002:
        raise RuntimeError(f"Graph/eager mismatch: {graph_check}")
    write_json(folder / "graph_parity.json", graph_check)
    rows = []
    for trial in range(args.trials):
        modes = ("eager", "cuda_graph") if trial % 2 == 0 else ("cuda_graph", "eager")
        for mode in modes:
            run = graph.replay if mode == "cuda_graph" else lambda: fn(source)
            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = gpu_status()
            result = measure(run, args.iterations)
            row = {"trial": trial + 1, "execution": mode, "before": before, "after": gpu_status(), "result": result}
            rows.append(row)
            write_json(folder / "trials.json", rows)
            print(f"{args.model} {width}x{args.height} {mode} {trial + 1}/{args.trials}: "
                  f"{result['median_ms']:.4f} ms; p95 {result['p95_ms']:.4f} ms", flush=True)
    summary = {"model": args.model, "input_shape_wh": [width, args.height], "results": {}}
    for mode in ("eager", "cuda_graph"):
        values = [row["result"] for row in rows if row["execution"] == mode]
        summary["results"][mode] = {key: statistics.median(v[key] for v in values)
                                    for key in ("median_ms", "p95_ms", "synchronized_loop_ms_per_frame",
                                                "peak_allocated_gib", "peak_reserved_gib")}
        summary["results"][mode]["median_range_ms"] = [min(v["median_ms"] for v in values), max(v["median_ms"] for v in values)]
    write_json(folder / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--height", type=int, choices=(540, 1080), default=540)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.model:
        worker(args)
        return
    for height in (540, 1080):
        for name in MODELS:
            subprocess.run([sys.executable, str(Path(__file__)), "--output", str(args.output), "--model", name,
                            "--height", str(height), "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                            "--trials", str(args.trials)], check=True)
    summaries = [json.loads(p.read_text()) for p in sorted((args.output / "pytorch").glob("*/summary.json"))]
    write_json(args.output / "pytorch_summary.json", summaries)


if __name__ == "__main__":
    main()
