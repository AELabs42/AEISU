"""Small deterministic CPU correctness checks; no external data or downloads."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from aeisu.data import PairedCrops, d4
from aeisu.metrics import distortion
from aeisu.model import AEISU, FoldedAEISU, fold_model
from aeisu.reconstruction import coefficient_clamp, project_basis, reconstruct, unpack_coefficients
from aeisu.runtime import load_model, upscale
from train import cosine_lr


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_coefficient_clamp(self):
        value = torch.tensor([-20., -12., -4., -2., 0., 2., 4., 12., 20.])
        expected = torch.tensor([-2.5, -2.5, -2.1, -2., 0., 2., 2.1, 2.5, 2.5])
        torch.testing.assert_close(coefficient_clamp(value), expected)

    def test_parity_packing(self):
        meta = torch.empty(1, 64, 2, 3)
        for parity in range(4):
            for component in range(16):
                meta[:, parity * 16 + component] = parity * .1 + component * .001
        coeff = unpack_coefficients(meta)
        for y in range(4):
            for x in range(6):
                expected = torch.arange(16) * .001 + (2 * (y % 2) + x % 2) * .1
                torch.testing.assert_close(coeff[0, :, y, x], expected)

    def test_basis_projection(self):
        raw = torch.randn(16, 4, 25, requires_grad=True)
        basis = project_basis(raw)
        torch.testing.assert_close(basis.sum(0), torch.zeros(16, 4), atol=5e-8, rtol=0)
        self.assertLessEqual(float(basis.detach().abs().sum(0).max()), .5000002)
        basis.square().sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        torch.testing.assert_close(project_basis(torch.zeros_like(raw)), torch.zeros(25, 16, 4))

    def test_initial_bilinear(self):
        model = AEISU()
        source = torch.rand(1, 3, 12, 16)
        prediction = model(source)
        expected = F.interpolate(source, scale_factor=2, mode="bilinear", align_corners=False)
        torch.testing.assert_close(prediction, expected, atol=3e-7, rtol=0)

    def test_folding_and_checkpoint(self):
        model = AEISU().eval()
        with torch.no_grad():
            model.metadata_head.weight.normal_(0, .02)
        source = torch.rand(1, 3, 16, 20)
        folded = fold_model(model)
        self.assertEqual(sum(parameter.numel() for parameter in folded.parameters()), 115996)
        torch.testing.assert_close(model(source), folded(source), atol=3e-6, rtol=3e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folded.pt"
            torch.save({"model": model.folded_production_state_dict()}, path)
            torch.testing.assert_close(load_model(path)(source), folded(source), atol=0, rtol=0)
            path = Path(directory) / "training.pt"
            torch.save({"model": model.state_dict()}, path)
            torch.testing.assert_close(load_model(path)(source), folded(source), atol=0, rtol=0)

    def test_sampled_gradients(self):
        source = torch.rand(1, 3, 8, 12)
        metadata = torch.randn(1, 64, 4, 6) * .2
        base = torch.randn(25, 4) * .01
        basis = torch.randn(25, 16, 4) * .01
        target = torch.rand(1, 3, 16, 24)
        gradients = []
        stride, phase = 4, 5
        for mode in ("dense", "sampled", "masked"):
            values = [value.clone().requires_grad_() for value in (metadata, base, basis)]
            prediction = reconstruct(source, *values, gradient_stride=stride if mode == "sampled" else 1, phase=phase)
            error = (prediction - target).square()
            if mode == "masked":
                mask = torch.zeros(8, 12)
                mask[phase // stride::stride, phase % stride::stride] = stride ** 2
                error = error * mask.repeat_interleave(2, 0).repeat_interleave(2, 1)
            error.mean().backward()
            gradients.append([value.grad for value in values])
        torch.testing.assert_close(gradients[0][0], gradients[1][0], atol=1e-7, rtol=1e-6)
        for index in (1, 2):
            torch.testing.assert_close(gradients[1][index], gradients[2][index], atol=2e-7, rtol=2e-6)

    def test_disabled_filter_gradients(self):
        model = AEISU()
        source = torch.rand(1, 3, 8, 8)
        model(source, gradient_stride=0).sum().backward()
        self.assertIsNone(model.base_kernel.grad)
        self.assertIsNone(model.basis_raw.grad)
        self.assertIsNotNone(model.metadata_head.weight.grad)

    def test_tiling_and_arbitrary_dimensions(self):
        model = FoldedAEISU().eval()
        with torch.no_grad():
            model.metadata_head.weight.normal_(0, .02)
            model.basis_raw.normal_(0, .05)
        source = torch.rand(3, 69, 83)
        whole, tiled = upscale(model, source, 0), upscale(model, source, 32)
        self.assertEqual(tiled.shape, (3, 138, 166))
        torch.testing.assert_close(tiled, whole, atol=2e-6, rtol=2e-6)

    def test_metrics_and_schedule(self):
        image = torch.rand(3, 20, 24)
        scores = distortion(image, image)
        self.assertEqual(scores["y_psnr"], float("inf"))
        self.assertAlmostEqual(scores["y_ssim"], 1, places=12)
        self.assertEqual(cosine_lr(0, 10, .002, 1e-6), .002)
        self.assertEqual(cosine_lr(9, 10, .002, 1e-6), 1e-6)

    def test_pair_crops_and_d4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            (root / "target").mkdir()
            source = np.random.default_rng(17).integers(0, 256, (32, 40, 3), dtype=np.uint8)
            target = source.repeat(2, 0).repeat(2, 1)
            Image.fromarray(source).save(root / "input/scene.png")
            Image.fromarray(target).save(root / "target/scene.png")
            dataset = PairedCrops(root / "input", root / "target", patch=32, samples=8)
            for index in range(8):
                a, b = dataset[index]
                core = a[:, 16:-16, 16:-16]
                torch.testing.assert_close(core.repeat_interleave(2, -2).repeat_interleave(2, -1), b, atol=0, rtol=0)
            tensor = torch.arange(9).reshape(1, 3, 3)
            self.assertEqual(len({tuple(d4(tensor, transform).flatten().tolist()) for transform in range(8)}), 8)

    def test_cpu_cli_workflow_and_resume(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "input").mkdir()
            (folder / "target").mkdir()
            source = np.random.default_rng(23).integers(0, 256, (32, 40, 3), dtype=np.uint8)
            target = source.repeat(2, 0).repeat(2, 1)
            Image.fromarray(source).save(folder / "input/scene.v1.png")
            Image.fromarray(target).save(folder / "target/scene.v1.png")
            config = json.loads((root / "configs/cpu_smoke.json").read_text())
            config["epochs"] = 2
            config_path = folder / "config.json"
            config_path.write_text(json.dumps(config))
            environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
            def run(script, *arguments):
                command = [sys.executable, str(root / script), *map(str, arguments)]
                result = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            common = ("--config", config_path, "--input", folder / "input", "--target", folder / "target", "--device", "cpu")
            run("train.py", *common, "--output", folder / "full")
            run("train.py", *common, "--output", folder / "resumed", "--stop-after-epoch", "1")
            run("train.py", *common, "--output", folder / "resumed", "--resume", folder / "resumed/last.pt")
            full = torch.load(folder / "full/last.pt", weights_only=True)
            resumed = torch.load(folder / "resumed/last.pt", weights_only=True)
            self.assertEqual(resumed["global_step"], 4)
            for name in full["model"]:
                torch.testing.assert_close(full["model"][name], resumed["model"][name], atol=0, rtol=0)
            run("export.py", "--checkpoint", folder / "full/last.pt", "--output", folder / "folded.pt")
            run("infer.py", "--checkpoint", folder / "folded.pt", "--input", folder / "input/scene.v1.png",
                "--output", folder / "result.png", "--tile", "32")
            with Image.open(folder / "result.png") as image:
                self.assertEqual(image.size, (80, 64))
            run("evaluate.py", "--checkpoint", folder / "folded.pt", "--input", folder / "input",
                "--target", folder / "target", "--output", folder / "evaluation", "--tile", "32")
            summary = json.loads((folder / "evaluation/summary.json").read_text())
            self.assertEqual(summary["pairs"], 1)
            self.assertTrue(np.isfinite(summary["mean_per_image"]["y_psnr"]))
            self.assertTrue((folder / "evaluation/images/scene.v1.npz").is_file())


if __name__ == "__main__":
    unittest.main()
