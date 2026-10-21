"""File-backed paired images, aligned random crops and D4 augmentation."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .runtime import IMAGE_EXTENSIONS, read_rgb


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def paired_images(source: Path, target: Path) -> list[tuple[Path, Path]]:
    def index(root):
        values = {}
        for path in list_images(root):
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in values:
                raise ValueError(f"Ambiguous relative image stem: {key}")
            values[key] = path
        return values
    a, b = index(source), index(target)
    if not a or a.keys() != b.keys():
        raise ValueError(f"Pair folders must have identical relative stems; inputs={len(a)}, targets={len(b)}")
    return [(a[key], b[key]) for key in sorted(a)]


def d4(image: torch.Tensor, transform: int) -> torch.Tensor:
    if transform >= 4:
        image = image.flip(-1)
    return torch.rot90(image, transform % 4, (-2, -1)).contiguous()


class PairedCrops(Dataset):
    def __init__(self, source, target, *, patch=512, context=16, samples=28000, seed=1234):
        if patch < 8 or patch % 8 or context < 2 or context % 4:
            raise ValueError("HR patch must be a positive multiple of 8; LR context a multiple of 4 >= 4")
        if samples < 1:
            raise ValueError("samples must be positive")
        self.pairs = paired_images(Path(source), Path(target))
        self.patch, self.context, self.samples, self.seed = patch, context, samples, seed
        self.epoch = 0

    def __len__(self):
        return self.samples

    @lru_cache(maxsize=2)
    def _read_pair(self, index):
        a, b = self.pairs[index]
        source, target = read_rgb(a), read_rgb(b)
        if target.shape[-2:] != tuple(value * 2 for value in source.shape[-2:]):
            raise ValueError(f"Pair is not aligned 2x geometry: {a.name}, {b.name}")
        return source, target

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * self.samples + index)
        pair = int(torch.randint(len(self.pairs), (), generator=generator))
        source, target = self._read_pair(pair)
        content = self.patch // 2
        h, w = source.shape[-2:]
        if h < content or w < content:
            raise ValueError("Training image is smaller than the requested crop")
        y = int(torch.randint(h - content + 1, (), generator=generator))
        x = int(torch.randint(w - content + 1, (), generator=generator))
        padding = self.context
        # Only pad a local window, avoiding another full-image allocation per crop.
        sy, sx = max(y - padding, 0), max(x - padding, 0)
        ey, ex = min(y + content + padding, h), min(x + content + padding, w)
        crop = source[:, sy:ey, sx:ex]
        crop = F.pad(crop[None], (max(padding - x, 0), max(x + content + padding - w, 0),
                                max(padding - y, 0), max(y + content + padding - h, 0)), mode="replicate")[0]
        reference = target[:, y * 2:(y + content) * 2, x * 2:(x + content) * 2]
        transform = int(torch.randint(8, (), generator=generator))
        return d4(crop, transform), d4(reference, transform)
