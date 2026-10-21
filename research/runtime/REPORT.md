# AEISU runtime comparison - 6 October 2026

## Common PyTorch

All primary rows use batch-one full-frame FP16, channels-last, cuDNN autotuning and TF32 disabled. Each shape/model runs in an isolated process; eager and CUDA graph order alternate across five trials. CUDA events measure resident input through clipped output, including dynamic padding, analysis and reconstruction. Uploads, setup and disk I/O are outside the interval. No custom CUDA extension or torch.compile is used.

| Model | 540p eager | 540p graph | 1080p eager | 1080p graph |
|---|---:|---:|---:|---:|
| AEISU | 1.721 (1.761) | 1.588 (1.592) | 7.064 (7.082) | 6.905 (6.917) |
| QuickSRNet S | 0.326 (0.331) | 0.292 (0.294) | 1.860 (1.868) | 1.839 (1.842) |
| QuickSRNet M | 0.611 (0.617) | 0.554 (0.558) | 3.545 (3.554) | 3.505 (3.513) |
| RepNet | 0.173 (0.244) | 0.129 (0.130) | 0.535 (0.541) | 0.514 (0.517) |
| RT4KSR | 1.108 (1.263) | 0.509 (0.511) | 1.968 (2.241) | 1.719 (1.725) |
| SPAN | 5.922 (5.956) | 5.820 (5.846) | 30.996 (31.192) | 30.884 (31.032) |
| SwinIR-Light | 124.587 (124.876) | 123.529 (123.811) | 499.289 (499.491) | 498.413 (498.857) |
| AEISU (FP32 synthesis) | 2.770 (2.788) | 2.661 (2.664) | 13.172 (13.199) | 13.022 (13.050) |

Values are milliseconds, median (p95), each aggregated across trials. AEISU's ordinary-Torch adapter folds fixed analysis into one convolution, packs branch pairs as grouped convolutions, and reconstructs through unfold/GEMM/batched-matmul. Its FP16 random-probe output agrees with the FP32 checkpoint at over 68 dB; this is numerical agreement, not a target-quality benchmark. The auxiliary row retains FP32 analysis/synthesis and is not a uniform-precision comparison.

## Optimized deployments

| Route | 540p to 1080p | 1080p to 4K |
|---|---:|---:|
| tensor_fp16 | 0.282 (0.302) | 0.975 (0.993) |
| tensor_mixed | 0.132 (0.152) | 0.418 (0.442) |
| shader_fp16 | 0.448 (0.450) | 1.679 (1.684) |
| shader_int8_fp16 | 0.334 (0.336) | 1.275 (1.279) |

This compares actual deployments, including different precision, fusion, layouts and instruction paths. It does not isolate Tensor Core hardware alone. Shader SASS is audited for zero matrix/tensor instructions.

## Native scaling repeat

| LR height | FP16 | uniform FP8 | mixed | mixed p95 |
|---|---:|---:|---:|---:|
| 540 | 0.285 | 0.204 | 0.132 | 0.152 |
| 720 | 0.473 | 0.308 | 0.206 | 0.211 |
| 1080 | 0.974 | 0.488 | 0.416 | 0.440 |
| 1440 | 1.677 | 1.132 | 0.740 | 0.764 |

Uniform FP8 scales 2.320x from 1080p to 1440p versus 1.778x logical pixels. The repeat reproduces nonlinear scaling, with stage growth in projection, refinement and the unfused coefficient head. Detailed events perturb execution and are not substituted for minimal-event total latency. The old 1080p mixed p95 of 0.678 ms is not reproduced; the fresh repeat is approximately 0.440 ms. Historical runs remain archived, with no unsupported attribution of the discrepancy to a single cause.

## Interpretation

AEISU is slower than QuickSRNet and RepNet in ordinary PyTorch. Its optimized deployment changes that ranking relative to QuickSRNet, while RepNet remains competitive in resident latency. The paper should present the optimized systems result and common-runtime result together, not infer an intrinsic architecture-only speed advantage. Quality remains degradation-dependent, and the latency-quality plot labels each backend. No output-head ablation performance claims are introduced.

PyTorch/spatial raw samples, native timing logs, graph/eager checks, checkpoint/source hashes, instruction audit and before/after hardware records are retained below this report directory.
