"""Synthetic behavior checks for the custom isolated-edge diagnostics."""
from __future__ import annotations

import json

import cv2
import numpy as np

from profiles import Protocol, profile_metrics, sample, select_edges


def main():
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    protocol = Protocol()
    size = 256
    xx = np.indices((size, size))[1]
    target = np.repeat(np.where(xx < size // 2, .2, .8).astype(np.float32)[..., None], 3, axis=2)
    edges = select_edges(target, "synthetic-step", protocol)
    assert len(edges["x"]) >= 10, "Clean synthetic step was not detected"
    scores = {}
    candidates = {"sharp": target, "blurred": cv2.GaussianBlur(target, (0, 0), 2.),
                  "biased": target * .9 + .04, "shifted": np.roll(target, 2, axis=1),
                  "constant": np.full_like(target, .5)}
    ring = target.copy()
    ring[:, size // 2 - 3:size // 2] -= .06
    ring[:, size // 2:size // 2 + 3] += .06
    candidates["ringing_10pct"] = ring
    for name, image in candidates.items():
        profiles = sample(image, edges["x"], edges["y"], edges["nx"], edges["ny"], protocol)
        scores[name], _ = profile_metrics(profiles, edges, protocol)
    for name in ("sharp", "blurred", "biased", "shifted", "constant"):
        for metric in ("bright_excursion_pct", "dark_excursion_pct", "ringing_excess_pct"):
            assert scores[name][metric] < 1e-3, (name, metric, scores[name][metric])
    assert scores["ringing_10pct"]["bright_excursion_pct"] > 9.9
    assert scores["ringing_10pct"]["dark_excursion_pct"] > 9.9
    assert scores["ringing_10pct"]["ringing_excess_pct"] > 19.9
    assert scores["blurred"]["width_10_90_px"] > scores["sharp"]["width_10_90_px"] + 3
    assert abs(scores["shifted"]["absolute_shift_50_px"] - 2) < .1
    assert scores["constant"]["width_defined_fraction"] == 0 and scores["constant"]["contrast_ratio"] == 0
    assert len(select_edges(np.full_like(target, .5), "flat", protocol)["x"]) == 0
    yy, xx = np.indices((size, size))
    curve = np.repeat(np.where((xx - 128)**2 + (yy - 128)**2 <= 80**2, .2, .8).astype(np.float32)[..., None], 3, axis=2)
    assert len(select_edges(curve, "curve", protocol)["x"]) > 10
    print(json.dumps({"passed": True, "step_edges": len(edges["x"]), "cases": scores}, indent=2))


if __name__ == "__main__":
    main()
