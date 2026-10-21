"""Extract verified, effective AEISU parameters on CPU; never run inference."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/paper_bench"))
from common import AESS, ARTIFACT, CHECKPOINT, RUN, sha256  # noqa: E402

sys.path.insert(0, str(AESS / "tools/srcpyr_fp8"))
from adaptive_reference import effective_base, effective_basis, load_adaptive  # noqa: E402
from model_format import load_artifact  # noqa: E402

PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
EXPECTED_CHECKPOINT = "d0208712e0d5e4a226335f6eddd4d926dd32f67de32ff590e836d219e99edefe"


def bilinear_base() -> np.ndarray:
    """Half-pixel x2 bilinear cross-correlation weights, [phase, y, x]."""
    offsets = np.arange(-2, 3, dtype=np.float64)
    phases = []
    for py, px in PHASES:
        wy = np.maximum(1.0 - np.abs(offsets - (-0.25 + 0.5 * py)), 0.0)
        wx = np.maximum(1.0 - np.abs(offsets - (-0.25 + 0.5 * px)), 0.0)
        phases.append(np.outer(wy, wx))
    result = np.asarray(phases, dtype=np.float32)
    assert np.all(result.sum((1, 2)) == 1.0)
    assert np.all(np.count_nonzero(result, axis=(1, 2)) == 4)
    assert np.all(result[:, 2, 2] == 9.0 / 16.0)
    return result


def stencils(parameters: list[np.ndarray]) -> np.ndarray:
    """Exact cross-correlation convention used by the training/runtime source."""
    (a, b, c), (d, e, f), (g, h, i), (j, k, ell) = parameters
    return np.asarray([
        [[-a, 0, a], [-b, 0, b], [-c, 0, c]],
        [[-d, -e, -f], [0, 0, 0], [d, e, f]],
        [[-g, -h, 0], [-i, 0, i], [0, h, g]],
        [[0, k, j], [-ell, 0, ell], [-j, -k, 0]],
    ], dtype=np.float32)


def validate_stencils(kernels: np.ndarray, parameters: list[np.ndarray]) -> float:
    """Compare matrix stencils with independent scalar source expressions."""
    patch = np.random.default_rng(29).normal(size=(3, 3))
    ul, uc, ur, cl, _, cr, dl, dc, dr = patch.ravel()
    dx, dy, d45, d135 = parameters
    expected = np.array([
        (ur - ul) * dx[0] + (cr - cl) * dx[1] + (dr - dl) * dx[2],
        (dl - ul) * dy[0] + (dc - uc) * dy[1] + (dr - ur) * dy[2],
        (dr - ul) * d45[0] + (dc - uc) * d45[1] + (cr - cl) * d45[2],
        (ur - dl) * d135[0] - (dc - uc) * d135[1] + (cr - cl) * d135[2],
    ])
    error = float(np.max(np.abs(np.einsum("kyx,yx->k", kernels, patch) - expected)))
    assert error < 1e-6, error
    return error


def extract() -> dict:
    if torch.version.cuda != "13.0":
        raise RuntimeError("Use the existing Torch CUDA 13.0 environment (CPU execution only).")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT or not (RUN / "final.txt").is_file():
        raise RuntimeError("Selected final checkpoint changed; review the paper's frozen model identity.")
    torch.set_num_threads(2)
    _, payload = load_adaptive(CHECKPOINT, torch.device("cpu"))
    if payload.get("folded_reparameterization") is not True:
        raise RuntimeError("This exporter requires the already-folded production checkpoint.")
    state = payload["model"]
    bank_tensor = effective_basis(payload, torch.device("cpu"))
    base_tensor = effective_base(payload, torch.device("cpu"))
    bank = bank_tensor.permute(1, 2, 0).reshape(16, 4, 5, 5).numpy()
    base = base_tensor.T.reshape(4, 5, 5).numpy()
    raw = state["basis_raw"].double().numpy()
    centered = raw - raw.mean(-1, keepdims=True)
    independently_projected = 0.5 * centered / np.maximum(np.abs(centered).sum(-1, keepdims=True), 1e-6)
    projection_error = float(np.max(np.abs(bank.reshape(16, 4, 25) - independently_projected)))
    assert projection_error < 1e-7
    assert np.max(np.abs(bank.sum((2, 3)))) < 1e-6
    assert np.max(np.abs(np.abs(bank).sum((2, 3)) - 0.5)) < 1e-6

    names = ["analysis_dx_rows", "analysis_dy_cols", "analysis_diag45", "analysis_diag135"]
    learned_vectors = [state[name].numpy() for name in names]
    original_vectors = [np.array(v, dtype=np.float32) for v in
                        ((0.125, 0.25, 0.125), (0.125, 0.25, 0.125), (0.25, 0.125, 0.125), (0.25, 0.125, 0.125))]
    learned_stencils, original_stencils = stencils(learned_vectors), stencils(original_vectors)
    stencil_check = max(validate_stencils(learned_stencils, learned_vectors),
                        validate_stencils(original_stencils, original_vectors))
    matrix = state["analysis_rgb_basis"].numpy()
    luma = state["analysis_rgb_mix"].numpy()
    initial_luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    artifact = load_artifact(ARTIFACT)
    flat_inputs = np.concatenate([matrix.ravel(), luma, *learned_vectors])
    native_checks = {}
    for name, value in (("adaptive_basis", bank_tensor.numpy()), ("adaptive_base", base_tensor.numpy()),
                        ("analysis_stencil", flat_inputs)):
        native_value = artifact.tensors[name].numpy()
        if native_value.shape != value.shape:
            raise RuntimeError(f"Native artifact shape changed for {name}")
        error = float(np.max(np.abs(native_value - value)))
        if error > 1e-7:
            raise RuntimeError(f"Native artifact differs for {name}: {error}")
        native_checks[name] = {"maximum_absolute_difference": error, "dtype": str(native_value.dtype)}
    arrays = {
        "basis_effective_k_phase_y_x": bank,
        "base_learned_phase_y_x": base,
        "base_bilinear_phase_y_x": bilinear_base(),
        "input_rgb_original": np.eye(3, dtype=np.float32),
        "input_rgb_learned": matrix,
        "input_luma_original": initial_luma,
        "input_luma_learned": luma,
        "input_stencils_original_direction_y_x": original_stencils,
        "input_stencils_learned_direction_y_x": learned_stencils,
        "input_composed_rgb_stencils_original_direction_channel_y_x": original_stencils[:, None] * initial_luma[None, :, None, None],
        "input_composed_rgb_stencils_learned_direction_channel_y_x": learned_stencils[:, None] * luma[None, :, None, None],
    }
    return {
        "date": "2026-09-29",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "epoch": payload["epoch"], "architecture": payload["architecture"],
        "native_artifact": str(ARTIFACT), "native_artifact_sha256": sha256(ARTIFACT),
        "final_marker": str(RUN / "final.txt"), "execution": "CPU only; no inference, retraining or GPU benchmark",
        "semantics": {
            "phases": [list(p) for p in PHASES], "phase_order": "TL, TR, BL, BR; (row, column)",
            "tap_order": "Rows: dy=-2..2 downward; columns: dx=-2..2 rightward; cross-correlation, no reversal",
            "basis": "Folded raw bank, mean-subtracted per basis/phase, then 0.5*centered/max(L1,1e-6)",
            "base": "Unnormalized learned folded base; original comparator is exact half-pixel bilinear x2",
            "inputs": "Folded effective parameters, not unfused reparameterization branches; z=M*RGB, y=m^T*RGB; directional stencils act on y",
            "interpretation": "Parameters and response mechanics only; no causal quality ablation or performance claim",
            "scales": "One symmetric scale per declared panel group, never per kernel. Bank and base have different explicitly labeled limits.",
        },
        "validation": {
            "basis_projection_max_error_vs_float64": projection_error,
            "basis_max_abs_dc": float(np.abs(bank.sum((2, 3))).max()),
            "basis_l1_range": [float(np.abs(bank).sum((2, 3)).min()), float(np.abs(bank).sum((2, 3)).max())],
            "base_phase_dc_sums": base.sum((1, 2), dtype=np.float64).tolist(),
            "stencil_expression_max_error": stencil_check,
            "native_parameter_checks": native_checks,
        },
        "source_hashes": {str(p): sha256(p) for p in (
            Path(__file__), ROOT / "Train_Core.py", AESS / "tools/srcpyr_fp8/adaptive_reference.py")},
        "arrays": {name: array.tolist() for name, array in arrays.items()},
        "array_sha256": {name: hashlib.sha256(array.astype("<f4").tobytes()).hexdigest() for name, array in arrays.items()},
    }
