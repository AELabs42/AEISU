# Coefficient-clamp checkpoint comparison

## Protocol

Both 100 epochs, 87500 updates, seed1234, area700 pairs, batch32, HR512, D4, MSE, cosine2e-3 to1e-6. clamped: fixed offsets, fused_cuda_16x5 (historical); unclamped: fixed offsets, fused_cuda_16x5. One training seed.

Common FP16 controller / FP32 analysis, constants and tiled reconstruction; align32 + halo16; clipped float32

All 600 frozen QRISP pairs and five established input conditions are used. Targets, selection, padding, folding, output clipping and metric definitions are shared. Old archive files are read-only. Both checkpoints use the same tiled FP32 reconstruction so the no-clamp model never encounters the fused runtime's hard-coded coefficient transfer.

Primary metrics are encoded BT.601 Y PSNR/SSIM after a two-HR-pixel crop and full-resolution cropped RGB LPIPS-Alex v0.1. RGB and full-frame results are retained in summary.json.

## Training loss

| Model | Final training MSE |
|---|---:|
| Soft coefficient clamp | 0.000114711334 |
| No coefficient clamp | 0.000116643196 |

## Reconstruction quality

| Input | Model | Y PSNR | Y SSIM | LPIPS |
|---|---|---:|---:|---:|
| ssaa_area | Soft coefficient clamp | 31.29696 | 0.9137457 | 0.1138951 |
| ssaa_area | No coefficient clamp | 31.25333 | 0.9133105 | 0.1151265 |
| bicubic_aa | Soft coefficient clamp | 31.03306 | 0.9070741 | 0.1519762 |
| bicubic_aa | No coefficient clamp | 31.00534 | 0.9068286 | 0.1529706 |
| fxaa | Soft coefficient clamp | 27.49601 | 0.8409153 | 0.1907038 |
| fxaa | No coefficient clamp | 27.49811 | 0.8409764 | 0.1914922 |
| smaa | Soft coefficient clamp | 27.14702 | 0.8500336 | 0.1718359 |
| smaa | No coefficient clamp | 27.15297 | 0.8501018 | 0.1728096 |
| native | Soft coefficient clamp | 26.68192 | 0.8472377 | 0.1618819 |
| native | No coefficient clamp | 26.69527 | 0.8475808 | 0.1631534 |

## Paired changes

No-clamp minus clamped. Positive PSNR/SSIM and negative LPIPS favor no-clamp. Exploratory hierarchical bootstrap bands share a 15-contrast family; only five environments are sampled; the bands do not measure training-seed variation.

| Input | Metric | Difference | Simultaneous 95% band |
|---|---|---:|---|
| ssaa_area | y_psnr_shave2_db | -0.04362593 | [-0.0725333, -0.01471857] |
| ssaa_area | y_ssim_shave2 | -0.0004352454 | [-0.0007097825, -0.0001607084] |
| ssaa_area | lpips_alex_v01 | +0.001231464 | [-0.0001258342, +0.002588761] |
| bicubic_aa | y_psnr_shave2_db | -0.02772394 | [-0.0481259, -0.007321966] |
| bicubic_aa | y_ssim_shave2 | -0.0002454325 | [-0.0004458244, -4.504072e-05] |
| bicubic_aa | lpips_alex_v01 | +0.0009944093 | [+1.447409e-05, +0.001974345] |
| fxaa | y_psnr_shave2_db | +0.002101626 | [-0.003109215, +0.007312467] |
| fxaa | y_ssim_shave2 | +6.106529e-05 | [-0.0001677171, +0.0002898476] |
| fxaa | lpips_alex_v01 | +0.0007883502 | [+5.264945e-05, +0.001524051] |
| smaa | y_psnr_shave2_db | +0.005950536 | [+4.59125e-05, +0.01185516] |
| smaa | y_ssim_shave2 | +6.814549e-05 | [-0.000175143, +0.000311434] |
| smaa | lpips_alex_v01 | +0.0009737185 | [-0.0003006723, +0.002248109] |
| native | y_psnr_shave2_db | +0.01334375 | [-0.002266065, +0.02895356] |
| native | y_ssim_shave2 | +0.0003430411 | [-0.0001587394, +0.0008448216] |
| native | lpips_alex_v01 | +0.001271464 | [-0.0009028943, +0.003445822] |

## Uniform-controller FP8 rounding stress on Area

Area-only E4M3 rounding simulation; per-output-channel weights, per-layer per-tensor calibrated inputs; FP32 arithmetic, no bias correction

Calibration uses the same 24 training crops, not evaluation images. This software stress test is not the production mixed FP8/FP16 model and is not a native runtime timing measurement.

| Model | FP8 PSNR | Delta PSNR | Delta SSIM | Delta LPIPS |
|---|---:|---:|---:|---:|
| Soft coefficient clamp | 31.25929 | -0.03766 | -0.0009532 | -0.0002404 |
| No coefficient clamp | 31.18402 | -0.06932 | -0.0011675 | +0.0011578 |

No-clamp minus clamped within-model precision changes:

| Metric | Difference | Simultaneous 95% band |
|---|---:|---|
| y_psnr_shave2_db | -0.03165082 | [-0.04870283, -0.0145988] |
| y_ssim_shave2 | -0.0002143611 | [-0.0004973909, +6.866867e-05] |
| lpips_alex_v01 | +0.001398271 | [+0.0001430827, +0.002653459] |

## Coefficient behavior

All unpadded LR sites; exact maxima, means and threshold fractions; deterministic 4096-value p99 sample per image

The raw range and applied range are different quantities. The soft transfer is identity to absolute value 2, has slope .05 to raw magnitude 12, and caps applied magnitude at 2.5. The p99 statistic is the mean of per-image sampled p99s, not a pooled whole-dataset quantile.

| Input | Model | Raw RMS | Mean raw p99 | Maximum raw magnitude | Fraction >2 | Applied RMS |
|---|---|---:|---:|---:|---:|---:|
| ssaa_area | Soft coefficient clamp | 1.02917 | 3.10278 | 19.18750 | 0.0506763 | 0.92995 |
| ssaa_area | No coefficient clamp | 2.63281 | 8.68343 | 63.90625 | 0.3572437 | 2.63281 |
| bicubic_aa | Soft coefficient clamp | 0.97267 | 2.83416 | 14.21094 | 0.0406459 | 0.90139 |
| bicubic_aa | No coefficient clamp | 2.47975 | 7.98174 | 58.43750 | 0.3421388 | 2.47975 |
| fxaa | Soft coefficient clamp | 0.97219 | 2.82560 | 16.12500 | 0.0407636 | 0.90235 |
| fxaa | No coefficient clamp | 2.49208 | 8.06309 | 53.06250 | 0.3429919 | 2.49208 |
| smaa | Soft coefficient clamp | 1.02386 | 3.07059 | 21.59375 | 0.0498894 | 0.92935 |
| smaa | No coefficient clamp | 2.62919 | 8.66851 | 77.50000 | 0.3576352 | 2.62919 |
| native | Soft coefficient clamp | 1.07034 | 3.27030 | 23.57812 | 0.0584974 | 0.95408 |
| native | No coefficient clamp | 2.73364 | 9.15392 | 86.93750 | 0.3692506 | 2.73364 |

## Interpretation limits

This observes the two trained checkpoints, not optimization difficulty directly. A smaller raw coefficient range does not by itself establish that coefficients must be more precise, and a quantization response also depends on trained controller/bank weights and calibration. Do not infer a general precision mechanism or distillation failure from these results alone.

Pre-output-clip RGB excursion fractions describe global [0,1] gamut violations, not local-edge haloing; legitimate transitions and ringing are not separated by this statistic. All 7200 float32 reconstructions and PNG previews are stored in this new results folder. No performance superiority claim is tested.
