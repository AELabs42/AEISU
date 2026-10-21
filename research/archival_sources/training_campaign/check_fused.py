"""Check soft/identity fused forward and sampled analytic gradients against tiled reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SHARED = Path(__file__).resolve().parents[2] / "ablations/_shared"
sys.path[:0] = [str(SHARED), str(SHARED / "snapshot")]
import torch
import adaptive_cuda_kernel as kernel
from reconstruction import coefficient_clamp, reconstruct, unpack_coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss", choices=("mse", "l1"), default="mse")
    args = parser.parse_args()
    if torch.version.cuda != "13.0":
        raise RuntimeError("Torch CUDA 13.0 required")
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for mode in ("soft", "none"):
        kernel.set_coefficient_transfer(mode)
        kernel.get_adaptive_cuda_module(raise_on_error=True)
        for phase in (0, 37):
            source = torch.rand(2, 3, 32, 40, device="cuda")
            target = torch.randint(0, 256, (2, 3, 64, 80), device="cuda", dtype=torch.uint8).contiguous(memory_format=torch.channels_last)
            meta = (torch.randn(2, 64, 16, 20, device="cuda") * 15).contiguous(memory_format=torch.channels_last).requires_grad_()
            base = (torch.randn(25, 4, device="cuda") * .05).requires_grad_()
            basis = (torch.randn(25, 16, 4, device="cuda") * .005).requires_grad_()
            fused_function = kernel.adaptive_mse_loss if args.loss == "mse" else kernel.adaptive_l1_loss
            fused = fused_function(source, meta, base, basis, target, 0, 16, phase)
            actual = torch.autograd.grad(fused, (meta, base, basis))
            coeff = coefficient_clamp(unpack_coefficients(meta, 1, 16, 1), mode)
            prediction = reconstruct(source, coeff, base, basis, 5, 0, 16, phase)
            reference_function = torch.nn.functional.mse_loss if args.loss == "mse" else torch.nn.functional.l1_loss
            reference = reference_function(prediction, target.float() / 255)
            expected = torch.autograd.grad(reference, (meta, base, basis))
            torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-5)
            errors = []
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=3e-3, atol=3e-5)
                errors.append(float((left - right).abs().max()))
            records.append({"mode": mode, "loss": args.loss, "stride": 16, "phase": phase,
                            "loss_error": float((fused - reference).abs()), "gradient_max_errors": errors})
    kernel.set_coefficient_transfer("soft")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "torch": torch.__version__, "cuda": torch.version.cuda,
                                       "tests": records}, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
