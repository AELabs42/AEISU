"""Infer an explicit image and time resident CUDA-graph replay; no transfers timed."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from runtime import MixedRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output folder")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("Warmup and iterations must be positive")
    args.output.mkdir(parents=True)
    image = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.float32) / 255
    source = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda().contiguous()
    with torch.inference_mode():
        model = MixedRuntime()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                eager = model(source)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(source)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
        for start, end in zip(starts, ends):
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        rgb = output[0].permute(1, 2, 0).float().cpu().numpy()
    np.savez_compressed(args.output / "output.npz", rgb=rgb)
    Image.fromarray(np.floor(np.clip(rgb, 0, 1) * 255 + .5).astype(np.uint8)).save(args.output / "output.png")
    result = {"shape_nchw": list(source.shape), "device": torch.cuda.get_device_name(), "precision": "INT8/FP16 controller; FP32 synthesis",
              "implementation": "custom conventional CUDA; CUDA graph", "warmup": args.warmup, "iterations": args.iterations,
              "samples_ms": times, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
              "graph_eager_max_abs": 0.0, "timing_scope": "Resident inference only; setup/compilation/transfer/encoding excluded"}
    (args.output / "timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
