"""Target-defined isolated-edge profile diagnostics, not a general quality metric."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import cv2
import numpy as np


@dataclass(frozen=True)
class Protocol:
    radius: int = 12
    sample_step: float = .5
    tangent_radius: int = 1
    plateau_start: int = 8
    grid: int = 16
    max_edges: int = 512
    smoothing_sigma: float = .8
    coherence_sigma: float = 2.
    minimum_gradient: float = .015
    minimum_coherence: float = .85
    minimum_contrast: float = .08
    maximum_plateau_spread: float = .06
    maximum_negative_variation: float = .12
    maximum_target_excursion: float = .08
    seed: int = 20261004

    def record(self):
        return asdict(self)


PRIMARY = ("bright_excursion_pct", "dark_excursion_pct", "ringing_excess_pct")
METRICS = PRIMARY + ("rgb_excursion_pct", "excursion_over2pct_fraction", "contrast_ratio",
                     "width_10_90_px", "width_defined_fraction", "absolute_shift_50_px")


def gray(rgb):
    # Encoded-RGB BT.601 luma. Offset/range of MATLAB Y cancel in contrast ratios.
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2GRAY)


def positions(protocol):
    return np.arange(-protocol.radius, protocol.radius + protocol.sample_step / 2, protocol.sample_step, dtype=np.float32)


def sample(rgb, x, y, nx, ny, protocol):
    """Bilinear profiles; identical sampling for target and all predictions."""
    steps = positions(protocol)
    samples = []
    for tangent in range(-protocol.tangent_radius, protocol.tangent_radius + 1):
        xx = x[:, None] + nx[:, None] * steps - ny[:, None] * tangent
        yy = y[:, None] + ny[:, None] * steps + nx[:, None] * tangent
        samples.append(cv2.remap(rgb, xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE))
    return np.stack(samples, axis=1)


def plateau_stats(profiles, protocol):
    steps = positions(protocol)
    left = profiles[:, :, steps <= -protocol.plateau_start]
    right = profiles[:, :, steps >= protocol.plateau_start]
    low = np.median(left, axis=(1, 2))
    high = np.median(right, axis=(1, 2))
    spread = np.maximum(np.ptp(left, axis=(1, 2)), np.ptp(right, axis=(1, 2)))
    return low, high, spread


def select_edges(target, identity, protocol):
    luma = gray(target)
    smooth = cv2.GaussianBlur(luma, (0, 0), protocol.smoothing_sigma, borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    strength = np.hypot(gx, gy)
    nx, ny = gx / np.maximum(strength, 1e-9), gy / np.maximum(strength, 1e-9)
    yy, xx = np.indices(luma.shape, dtype=np.float32)
    before = cv2.remap(strength, xx - nx, yy - ny, cv2.INTER_LINEAR)
    after = cv2.remap(strength, xx + nx, yy + ny, cv2.INTER_LINEAR)
    blur = lambda value: cv2.GaussianBlur(value, (0, 0), protocol.coherence_sigma)
    a, b, c = blur(gx * gx), blur(gy * gy), blur(gx * gy)
    coherence = np.sqrt((a - b)**2 + 4 * c**2) / np.maximum(a + b, 1e-9)
    margin = protocol.radius + protocol.tangent_radius + 2
    eligible = ((strength >= protocol.minimum_gradient) & (strength >= before) & (strength > after)
                & (coherence >= protocol.minimum_coherence)
                & (xx >= margin) & (yy >= margin) & (xx < luma.shape[1] - margin) & (yy < luma.shape[0] - margin))
    y, x = np.nonzero(eligible)
    if not len(x):
        return {"x": x, "y": y, "eligible_before_cap": 0, "candidates": 0}
    # One strongest target gradient per spatial cell, independent of model outputs.
    cells = (y // protocol.grid) * ((luma.shape[1] + protocol.grid - 1) // protocol.grid) + x // protocol.grid
    ordering = np.lexsort((y * luma.shape[1] + x, -strength[y, x], cells))
    _, first = np.unique(cells[ordering], return_index=True)
    chosen = ordering[first]
    x, y = x[chosen].astype(np.float32), y[chosen].astype(np.float32)
    normals_x, normals_y = nx[y.astype(int), x.astype(int)], ny[y.astype(int), x.astype(int)]
    profiles = sample(luma, x, y, normals_x, normals_y, protocol)
    low, high, spread = plateau_stats(profiles, protocol)
    contrast = high - low
    denominator = np.maximum(contrast, 1e-6)
    average = profiles.mean(axis=1)
    negative = np.maximum(-np.diff(average, axis=1), 0).sum(axis=1) / denominator
    target_excursion = np.maximum(average.max(axis=1) - high, low - average.min(axis=1)) / denominator
    good = ((contrast >= protocol.minimum_contrast) & (spread / denominator <= protocol.maximum_plateau_spread)
            & (negative <= protocol.maximum_negative_variation) & (target_excursion <= protocol.maximum_target_excursion))
    selected = np.flatnonzero(good)
    count = len(selected)
    if count > protocol.max_edges:
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") ^ protocol.seed
        selected = np.sort(np.random.default_rng(seed).choice(selected, protocol.max_edges, replace=False))
    if not len(selected):
        return {"x": x[:0], "y": y[:0], "eligible_before_cap": 0, "candidates": len(x)}
    return {"x": x[selected], "y": y[selected], "nx": normals_x[selected], "ny": normals_y[selected],
            "target_profiles": sample(target, x[selected], y[selected], normals_x[selected], normals_y[selected], protocol),
            "target_low": low[selected], "target_high": high[selected], "contrast": contrast[selected],
            "eligible_before_cap": count, "candidates": len(x)}


def luma_profiles(rgb_profiles):
    return np.einsum("...c,c->...", rgb_profiles.astype(np.float64), [.299, .587, .114])


def nearest_rising_crossing(profile, level, steps, reference):
    result = np.full(len(profile), np.nan)
    for index, row in enumerate(profile):
        choices = np.flatnonzero((row[:-1] <= level) & (row[1:] > level))
        if len(choices):
            crossing = steps[choices] + (level - row[choices]) * np.diff(steps)[choices] / (row[choices + 1] - row[choices])
            result[index] = crossing[np.argmin(np.abs(crossing - reference[index]))]
    return result


def profile_metrics(prediction_profiles, edges, protocol):
    target_profiles = edges["target_profiles"]
    pred = luma_profiles(prediction_profiles)
    target = luma_profiles(target_profiles)
    pl, ph, _ = plateau_stats(pred, protocol)
    tl, th, _ = plateau_stats(target, protocol)
    contrast = th - tl
    p, t = pred.mean(axis=1), target.mean(axis=1)
    bright = np.maximum(p.max(axis=1) - np.maximum(pl, ph), 0)
    dark = np.maximum(np.minimum(pl, ph) - p.min(axis=1), 0)
    tb = np.maximum(t.max(axis=1) - th, 0)
    td = np.maximum(tl - t.min(axis=1), 0)
    # Subtract the target's own excursions; a monotone but biased edge is not a halo.
    bright = np.maximum(bright - tb, 0) / contrast
    dark = np.maximum(dark - td, 0) / contrast
    variation = lambda rows: np.abs(np.diff(rows, axis=1)).sum(axis=1) - np.abs(rows[:, -1] - rows[:, 0])
    ringing = np.maximum(variation(p) - variation(t), 0) / (2 * contrast)
    prgb, trgb = prediction_profiles.mean(axis=1).astype(np.float64), target_profiles.mean(axis=1).astype(np.float64)
    plo, phi, _ = plateau_stats(prediction_profiles.astype(np.float64), protocol)
    tlo, thi, _ = plateau_stats(target_profiles.astype(np.float64), protocol)
    excursion = lambda rows, left, right: np.maximum(np.maximum(rows.max(axis=1) - np.maximum(left, right),
                                                               np.minimum(left, right) - rows.min(axis=1)), 0)
    rgb = np.maximum(excursion(prgb, plo, phi) - excursion(trgb, tlo, thi), 0).max(axis=1) / contrast
    steps = positions(protocol).astype(np.float64)
    tq = (t - tl[:, None]) / contrast[:, None]
    t50 = nearest_rising_crossing(tq, .5, steps, np.zeros(len(t)))
    # Own plateau normalization isolates transition width from brightness/gain shifts.
    pq = (p - pl[:, None]) / np.maximum((ph - pl)[:, None], 1e-8)
    p50 = nearest_rising_crossing(pq, .5, steps, t50)
    p10 = nearest_rising_crossing(pq, .1, steps, p50)
    p90 = nearest_rising_crossing(pq, .9, steps, p50)
    valid_width = (np.isfinite(p10) & np.isfinite(p90) & (p10 <= p50) & (p90 >= p50) & (ph - pl >= .25 * contrast))
    width = np.where(valid_width, p90 - p10, np.nan)
    shift = np.where(np.isfinite(p50) & (ph - pl >= .25 * contrast), np.abs(p50 - t50), np.nan)
    values = {"bright_excursion_pct": bright * 100, "dark_excursion_pct": dark * 100,
              "ringing_excess_pct": ringing * 100, "rgb_excursion_pct": rgb * 100,
              "excursion_over2pct_fraction": np.maximum(bright, dark) > .02,
              "contrast_ratio": (ph - pl) / contrast, "width_10_90_px": width,
              "width_defined_fraction": valid_width, "absolute_shift_50_px": shift}
    summary = {key: float(np.mean(value[np.isfinite(value)])) if np.isfinite(value).any() else None for key, value in values.items()}
    return summary, values
