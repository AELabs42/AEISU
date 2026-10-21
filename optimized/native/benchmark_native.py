"""Run the isolated native harness on an explicit RGB input, with archived logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "weights/adaptive_fp8.bin")
    parser.add_argument("--exe", type=Path, default=ROOT / "build/aeisu_native_bench.exe")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--save-output", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a fresh timing output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.float32) / 255
    h, w = rgb.shape[:2]
    source = args.output / "source_rgba_f32.bin"
    np.concatenate((rgb, np.ones((h, w, 1), np.float32)), axis=2).astype("<f4").tofile(source)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AESS_")}
    paths = args.exe.parent / "runtime_paths.json"
    if paths.is_file():
        values = json.loads(paths.read_text())
        environment["PATH"] = os.pathsep.join([values["cuda_bin"], values["cudnn_bin"], environment.get("PATH", "")])
    command = [str(args.exe.resolve()), "--model", str(args.model.resolve()), "--width", str(w), "--height", str(h),
               "--input", str(source.resolve()), "--dump-dir", str(args.output.resolve()), "--warmup", str(args.warmup),
               "--iterations", str(args.iterations), "--timing-mode", "detailed" if args.detailed else "minimal",
               "--output-only" if args.save_output else "--no-dump", "--min-psnr", "0"]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    (args.output / "native.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    source.unlink()
    protocol = {"input_shape_wh": [w, h], "batch": 1, "warmup": args.warmup, "iterations": args.iterations,
                "timing": "Resident analysis/controller/reconstruction, native CUDA events; setup/transfers outside interval",
                "mode": "detailed" if args.detailed else "minimal", "exit_code": result.returncode,
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "exe_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest()}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError(result.stderr)


if __name__ == "__main__":
    main()
