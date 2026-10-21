"""Export only folded model tensors and non-personal metadata."""
import argparse
from pathlib import Path

import torch

from aeisu.runtime import load_model, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-input", choices=("area", "bicubic", "unspecified"), default="unspecified")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.checkpoint.resolve():
        raise FileExistsError("Choose a new output path")
    model = load_model(args.checkpoint, "cpu")
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    payload = {
        "format_version": 1, "architecture": "aeisu_parallel32x4_folded_v1",
        "model": {name: value.cpu() for name, value in model.state_dict().items()},
        "basis_l1_scale": model.basis_l1_scale, "training_input": args.training_input,
        "source_sha256": sha256(args.checkpoint),
        "epoch": original.get("epoch"), "updates": original.get("global_step"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"Exported {sum(value.numel() for value in payload['model'].values()):,} folded values; SHA256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
