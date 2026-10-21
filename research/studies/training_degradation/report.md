# AEISU training-degradation comparison

Two main-architecture checkpoints, 1000 epochs and 875000 updates each. Shared 700 training targets, seed 1234, batch 32, 512 HR crops, D4, RGB MSE only and cosine 2e-3 to 1e-6. Complete optimizer/TensorBoard traces and target hashes are checked in training_audit.json.

Area uses the historical training inputs. Bicubic uses MATLAB-style antialiased 8K-to-1080p inputs with the same existing 4K targets. The QRISP Bicubic evaluation input is Pillow Bicubic, not identical to the training filter. Both checkpoints are CPU-FP32 folded and evaluated through common FP16 controller/FP32 constants and reconstruction with align32/halo16 padding. This study does not replace the published mixed-precision production model scores.

Evaluation reuses all 600 frozen QRISP pairs across five conditions and unchanged Native targets. PSNR/SSIM use encoded Y with two-HR-pixel cropping; LPIPS AlexNet v0.1 uses full-resolution cropped RGB. Full-frame and RGB metrics are retained. All 6000 float32 outputs and PNG previews are saved.

## Training losses

| Training input | Final MSE |
|---|---:|
| Area-trained | 0.000113809788 |
| Bicubic-trained | 0.000108882359 |

Training losses refer to different input distributions and are not held-out score comparisons.

## Held-out reconstruction

| Input | Training | Y PSNR | Y SSIM | LPIPS |
|---|---|---:|---:|---:|
| ssaa_area | Area-trained | 31.34560 | 0.9144562 | 0.1123844 |
| ssaa_area | Bicubic-trained | 30.60584 | 0.9087528 | 0.0986394 |
| bicubic_aa | Area-trained | 31.05964 | 0.9077551 | 0.1514577 |
| bicubic_aa | Bicubic-trained | 31.72505 | 0.9166408 | 0.1206653 |
| fxaa | Area-trained | 27.49622 | 0.8411096 | 0.1901509 |
| fxaa | Bicubic-trained | 27.34812 | 0.8385574 | 0.1752178 |
| smaa | Area-trained | 27.14127 | 0.8501020 | 0.1710733 |
| smaa | Bicubic-trained | 26.76566 | 0.8434124 | 0.1613048 |
| native | Area-trained | 26.67404 | 0.8473217 | 0.1610380 |
| native | Bicubic-trained | 25.95559 | 0.8338648 | 0.1635497 |

## Paired changes

Bicubic-trained minus Area-trained. Positive PSNR/SSIM and negative LPIPS favor Bicubic training. The 15 contrasts share simultaneous paired hierarchical-bootstrap bands.

| Input | Metric | Change | Simultaneous 95% band |
|---|---|---:|---|
| Bicubic-trained minus Area-trained / ssaa_area | y_psnr_shave2_db | -0.73976 | [-0.985066, -0.494455] |
| Bicubic-trained minus Area-trained / ssaa_area | y_ssim_shave2 | -0.00570346 | [-0.00874015, -0.00266678] |
| Bicubic-trained minus Area-trained / ssaa_area | lpips_alex_v01 | -0.013745 | [-0.0227198, -0.00477018] |
| Bicubic-trained minus Area-trained / bicubic_aa | y_psnr_shave2_db | +0.665405 | [+0.329526, +1.00128] |
| Bicubic-trained minus Area-trained / bicubic_aa | y_ssim_shave2 | +0.00888575 | [+0.00534284, +0.0124287] |
| Bicubic-trained minus Area-trained / bicubic_aa | lpips_alex_v01 | -0.0307925 | [-0.0347809, -0.026804] |
| Bicubic-trained minus Area-trained / fxaa | y_psnr_shave2_db | -0.148092 | [-0.211595, -0.0845895] |
| Bicubic-trained minus Area-trained / fxaa | y_ssim_shave2 | -0.00255221 | [-0.00433651, -0.00076791] |
| Bicubic-trained minus Area-trained / fxaa | lpips_alex_v01 | -0.0149331 | [-0.0219181, -0.00794803] |
| Bicubic-trained minus Area-trained / smaa | y_psnr_shave2_db | -0.375608 | [-0.476272, -0.274944] |
| Bicubic-trained minus Area-trained / smaa | y_ssim_shave2 | -0.00668961 | [-0.00923953, -0.0041397] |
| Bicubic-trained minus Area-trained / smaa | lpips_alex_v01 | -0.00976852 | [-0.0212493, +0.00171227] |
| Bicubic-trained minus Area-trained / native | y_psnr_shave2_db | -0.718448 | [-0.873242, -0.563653] |
| Bicubic-trained minus Area-trained / native | y_ssim_shave2 | -0.013457 | [-0.0209543, -0.00595961] |
| Bicubic-trained minus Area-trained / native | lpips_alex_v01 | +0.00251172 | [-0.00844609, +0.0134695] |

## Interpretation and limits

Direction, magnitude and paired support are recomputed in the tables above and summary.json. Earlier checkpoints' numerical rankings are not carried forward into this report. Distortion and perceptual metrics assess different properties; neither establishes universal rendering fidelity.

Same 700 target identities, budget and seed. Gradient policies: area_trained=fixed, bicubic_trained=fixed. Area input preparation and direct 8K MATLAB-style Bicubic differ; QRISP evaluation Bicubic is Pillow antialiased Bicubic.

One seed per training distribution; scene/sequence bands do not quantify retraining variation. All sampled environments/strata are retained, not selected according to resulting scores.

Independent verification: verification.json checks complete slot coverage and output presence, with first/last hash, decoded-array, PNG-rounding and metric checks in all ten slots.
