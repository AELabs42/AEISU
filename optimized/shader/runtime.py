"""Conventional-CUDA deployment of the exported profile-selected INT8/FP16 model."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent


def load_extension():
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("These kernels are validated only on RTX 5090 (SM 12.0)")
    if torch.version.cuda != "13.0":
        raise RuntimeError("Use the documented PyTorch CUDA 13.0 build")
    if os.name == "nt":
        sys.path.insert(0, str(ROOT.parent / "native"))
        from build_native import developer_environment
        os.environ.update(developer_environment())
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ["MAX_JOBS"] = "2"
    return load(name="aeisu_release_simt_v8", sources=[str(ROOT / "src/fusion_kernels.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-Xcompiler", "/Zc:preprocessor"],
                extra_cflags=["/O2", "/Zc:preprocessor"], verbose=True)


class MixedRuntime:
    """Exact saved mixed policy; tensor operations only prepare static weights."""
    integer_names = ("compress", "bottleneck0", "bottleneck1", "bottleneck2", "bottleneck3", "expand")

    def __init__(self, checkpoint=ROOT / "weights/aeisu_int8_fp16.pt", extension=None):
        record = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = {"project": "fp16", **{name: "int8" for name in self.integer_names}, "refine": "fp16", "tail": "fp16"}
        if record.get("format") != "shader_precision_v1" or record["precision_policy"] != expected or record["all_int8"]:
            raise ValueError("This runner accepts only the published mixed INT8/FP16 policy")
        self.extension = extension if extension is not None else load_extension()
        self.scales, self.paired_body, self.head_block = record["activation_scales"], record["paired_body"], record["head_block"]
        weights = {name: value.cuda().contiguous(memory_format=torch.channels_last) for name, value in record["half_weights"].items()}
        self.packed = {name: value.permute(2, 3, 1, 0).contiguous() for name, value in weights.items()}
        self.biases = {name: value.cuda() for name, value in record["half_biases"].items()}
        self.integer = {}
        for name, value in record["int8_weights"].items():
            co, ci, kh, kw = value.shape
            packed = value.cuda().permute(2, 3, 0, 1).contiguous().view(kh, kw, co, ci // 4, 4)
            self.integer[name] = packed.permute(0, 1, 3, 2, 4).contiguous().view(torch.int32).flatten()
        self.factors = {name: value.cuda() for name, value in record["int8_factors"].items()}
        self.integer_biases = {name: value.cuda() for name, value in record["int8_biases"].items()}
        self.analysis, self.base, self.bank = (record[name].cuda().contiguous() for name in ("analysis_parameters", "base", "bank"))
        transform = torch.tensor([[1., 0., 0.], [.5, .5, .5], [.5, -.5, .5], [0., 0., 1.]], device="cuda")
        self.refine = (transform @ weights["refine"].float() @ transform.T).permute(2, 3, 1, 0).half().contiguous()

    def __call__(self, source):
        if source.ndim != 4 or source.shape[:2] != (1, 3) or source.dtype != torch.float32 or not source.is_cuda:
            raise ValueError("Expected batch-one CUDA FP32 NCHW RGB")
        h, w = source.shape[-2:]
        py, px = 16 + (-h % 32) // 2, 16 + (-w % 32) // 2
        source = source.contiguous()
        skip = self.extension.project_source(source, self.analysis, self.packed["project"], self.biases["project"],
                                             h + 32 + (-h % 32), w + 32 + (-w % 32), py, px, 8)
        value = skip
        for index, name in enumerate(self.integer_names, start=1):
            if self.paired_body and index in (3, 5):
                continue
            if self.paired_body and index in (2, 4):
                next_name = self.integer_names[index]
                value = self.extension.quantized_pair(value, self.integer[name], self.integer[next_name],
                                                      self.factors[name], self.factors[next_name],
                                                      self.integer_biases[name], self.integer_biases[next_name],
                                                      self.scales[index + 1], self.scales[index + 2])
                continue
            kind = 0 if name == "compress" else 2 if name == "expand" else 1
            value = self.extension.quantized_body(value, skip, self.integer[name], self.factors[name], self.integer_biases[name],
                                                   kind, self.scales[index], self.scales[index + 1], 8)
        metadata = self.extension.finish_controller(value, self.refine, self.biases["refine"], self.packed["tail"], self.biases["tail"], 64)
        return self.extension.shared_head(source, metadata, self.base, self.bank, py, px, 0, self.head_block)
