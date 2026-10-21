"""Generic file-backed AEISU training; CPU FP32 or standard CUDA BF16 AMP."""
import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from aeisu.data import PairedCrops
from aeisu.model import AEISU
from aeisu.reconstruction import reconstruct
from aeisu.runtime import resolve_device


def cosine_lr(step, total, start, minimum):
    if total == 1:
        return start
    return minimum + .5 * (start - minimum) * (1 + math.cos(math.pi * step / (total - 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/area.json")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patch", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop at an epoch boundary without changing the planned cosine budget")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("epochs", "steps_per_epoch", "batch", "patch"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
        if int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if cfg["gradient_phase"] not in ("fixed", "cycle") or cfg["basis_grad_stride"] < 0:
        raise ValueError("Invalid gradient-sampling policy")
    if cfg["lr"] <= 0 or cfg["min_lr"] <= 0 or cfg["min_lr"] > cfg["lr"]:
        raise ValueError("Invalid learning-rate range")
    device = resolve_device(args.device)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Choose an empty training folder, or explicitly resume")
    dataset = PairedCrops(args.input, args.target, patch=cfg["patch"], context=cfg["source_pad"],
                          samples=cfg["steps_per_epoch"] * cfg["batch"], seed=seed)
    loader = DataLoader(dataset, batch_size=cfg["batch"], num_workers=args.workers, shuffle=False)
    model = AEISU(basis_l1_scale=cfg["basis_l1_scale"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(.9, .999), eps=1e-8, weight_decay=0)
    start_epoch, step = 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["config"] != cfg:
            raise ValueError("Resume requires the same complete training configuration")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, step = saved["epoch"], saved["global_step"]
    use_amp = device == "cuda" and cfg["amp"] == "bf16" and not args.no_amp
    if use_amp and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unavailable; use --no-amp")
    total = cfg["epochs"] * cfg["steps_per_epoch"]
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("stop-after-epoch must be positive")
    end_epoch = min(cfg["epochs"], args.stop_after_epoch or cfg["epochs"])
    if step != start_epoch * cfg["steps_per_epoch"] or start_epoch > cfg["epochs"]:
        raise ValueError("Resume checkpoint is not an epoch-boundary checkpoint for this recipe")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"AEISU | {device} | {'BF16 controller AMP' if use_amp else 'FP32'} | {len(dataset.pairs)} pairs", flush=True)
    log_path = args.output / "loss.csv"
    with log_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if not args.resume or log_path.stat().st_size == 0:
            writer.writerow(["epoch", "step", "mse", "learning_rate"])
        for epoch in range(start_epoch, end_epoch):
            dataset.epoch = epoch
            model.train()
            losses = []
            for source, target in loader:
                source, target = source.to(device), target.to(device)
                learning_rate = cosine_lr(step, total, cfg["lr"], cfg["min_lr"])
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    metadata = model.predict_coefficients(source)
                phase = step % max(1, cfg["basis_grad_stride"] ** 2) if cfg["gradient_phase"] == "cycle" else 0
                prediction = reconstruct(source, metadata, model.effective_base(), model.effective_basis(),
                                         trim=cfg["source_pad"] * 2, gradient_stride=cfg["basis_grad_stride"], phase=phase)
                loss = F.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                value = float(loss.detach())
                losses.append(value)
                writer.writerow([epoch + 1, step, value, learning_rate])
                if step % cfg["log_every"] == 0:
                    print(f"epoch={epoch + 1}/{cfg['epochs']} step={step}/{total} mse={value:.7g} lr={learning_rate:.7g}", flush=True)
            stream.flush()
            checkpoint = {"format_version": 1, "architecture": "aeisu_parallel32x4_training_v1",
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg,
                          "epoch": epoch + 1, "global_step": step, "metrics": {"train_mse": float(np.mean(losses))},
                          "execution": {"device": device, "controller_amp": "bf16" if use_amp else "fp32"}}
            temporary = args.output / "last.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "last.pt")
            print(f"epoch={epoch + 1} mean_mse={np.mean(losses):.7g}; saved last.pt", flush=True)


if __name__ == "__main__":
    main()
