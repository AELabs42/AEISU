# AEISU model card

AEISU is a single-frame, 2x RGB upscaler for already antialiased, tone-mapped, perceptually encoded input.
It predicts coefficients for a learned local reconstruction filter, not a neural RGB image.
The thirteen-convolution parallel controller operates below input resolution; synthesis uses the original RGB neighborhood.
The model does not itself provide a complete antialiasing, HDR tone-mapping or temporal reconstruction pipeline.

## Included models

- weights/aeisu_area.pt: folded full-precision principal Area model.
- weights/aeisu_bicubic.pt: folded full-precision training-degradation control.
- optimized/native/weights/adaptive_fp8.bin: production mixed FP8/FP16 deployment artifact.
- optimized/shader/weights/aeisu_int8_fp16.pt: conventional-CUDA profile-selected mixed INT8/FP16 artifact.

The portable checkpoints contain model tensors and non-personal provenance, not the original optimizer or training captures.
Weight source and packaged hashes are recorded in the release manifests. The two folded Python checkpoints support CPU inference.
Hardware-specific artifacts are not interchangeable with those checkpoints.

## Training and evidence

Training used 700 selected UI-free game frames (635 Honkai: Star Rail and 65 Red Dead Redemption 2) derived from compressed 8K captures, MSE,
D4 augmentation and a 0.002-to-0.000001 cosine schedule. The supplied principal models use final epoch-1000 checkpoints,
without internal-validation selection. Historical acquisition and scene-level split metadata are incomplete.
The portable scripts preserve the stated recipe but do not reproduce the original fused runtime arithmetic or exact random crop sequence.

The paper reports separately acquired QRISP evaluation across Area, FXAA, SMAA, Bicubic and Native input tracks,
with PSNR, SSIM and LPIPS and a source-pair selection frozen before comparing model outputs.
Area results do not imply universal restoration superiority. Native PSNR deficits and bicubic-domain weaknesses remain documented.
Visual artifact observations are accompanied by scoped isolated-edge diagnostics; small LPIPS gaps are not treated as resolved wins.
See paper/ and research/ for the full protocols, limitations, ablations, uncertainty estimates and implementation-specific latency evidence.

## Limitations and rights

Evidence is single-frame and the principal training bank is dominated by one game; temporal stability, other GPUs and broader capture distributions
are not established. Native inputs can contain aliasing or render differences absent from the target.
Model predictions are not ground truth or guaranteed lossless reconstruction.

Private training data and third-party comparator weights are not redistributed.
The author's code and released model weights are MIT-licensed. The manuscript is CC BY 4.0, excluding third-party imagery.
See LICENSE_SCOPE.md for the precise scope; third-party rights and upstream license terms remain unchanged.
No support for safety-critical uses is claimed.

The additional `weights/aeisu_div2k_l1.pt` model uses DIV2K bicubic pairs and L1 loss for 3,000 epochs (915,000 updates).
