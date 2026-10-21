"""Upscale a file or image tree using portable full-precision AEISU."""
import argparse
from pathlib import Path

from aeisu.data import list_images
from aeisu.runtime import load_model, read_rgb, resolve_device, save_rgb, upscale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--tile", type=int, default=128, help="LR core size, multiple of 32; zero for full frame")
    args = parser.parse_args()
    model = load_model(args.checkpoint, resolve_device(args.device))
    images = [args.input] if args.input.is_file() else list_images(args.input)
    if not images:
        raise ValueError("No input images found")
    for path in images:
        destination = args.output if args.input.is_file() else args.output / path.relative_to(args.input).with_suffix(".png")
        if path.resolve() == destination.resolve() or destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")
        save_rgb(destination, upscale(model, read_rgb(path), args.tile))
        print(f"{path.name} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
