# AEISU core architecture ablations

## Protocol

Four 100-epoch checkpoints, each with 87,500 updates, batch 32, 512x512 HR crops, D4 augmentation, RGB MSE and cosine 2e-3 to 1e-6. The controls change one factor each, not cumulatively.

The frozen QRISP sample has 600 pairs from five environments and three equally represented source-pair agreement strata. Every input condition uses the unchanged Native HR target. The Area and Bicubic inputs are target-derived; Native/FXAA/SMAA use the independently rendered LR. Neither stratum sorting nor these ablations identifies rendering mismatch as the sole cause of any score difference.

Primary quality: FP16 controller convolution execution with FP32 analysis/filter constants and fused reconstruction, no temporal information. All training checkpoints are folded on CPU in FP32 before evaluation. The established symmetric align32 and replicate-halo16 validation padding is shared by all models. PSNR/SSIM use encoded BT.601 Y, two-HR-pixel crop and FP64 arithmetic. LPIPS uses full-resolution cropped RGB, AlexNet v0.1 and FP32. Full-frame and RGB metrics are retained in summary.json. All 14,400 float32 reconstructions and original-color PNG previews are saved.

Training protocol: Gradient policies: reference=fixed, fixed_inputs=fixed, fixed_base=fixed, dense_core=fixed. One training seed; scene uncertainty is not seed uncertainty.

## Final training loss

| Model | MSE | Folded scalar count |
|---|---:|---:|
| Original 100-epoch reference | 0.000114711334 | 115,996 |
| Fixed inputs | 0.000115064486 | 115,996 |
| Fixed bilinear base | 0.000115658612 | 115,996 |
| Two dense 64-channel convolutions | 0.000115777409 | 115,868 |

Fixed-input and fixed-base controls retain the same deployed operator shapes as the reference; they change learned constants, not stage count. The dense core matches bottleneck convolution MACs but differs in depth, bias count and optimization. A separate generic same-backend timing diagnostic, where available, is reported below rather than pooled with the quality study.

## Quality: ssaa_area

| Model | Y PSNR (dB) | Y SSIM | LPIPS (lower is better) |
|---|---:|---:|---:|
| Original 100-epoch reference | 31.29696 | 0.9137457 | 0.1138950 |
| Fixed inputs | 31.30371 | 0.9139504 | 0.1121906 |
| Fixed bilinear base | 31.28310 | 0.9134560 | 0.1139476 |
| Two dense 64-channel convolutions | 31.27672 | 0.9137001 | 0.1139591 |

## Quality: bicubic_aa

| Model | Y PSNR (dB) | Y SSIM | LPIPS (lower is better) |
|---|---:|---:|---:|
| Original 100-epoch reference | 31.03306 | 0.9070741 | 0.1519762 |
| Fixed inputs | 31.05673 | 0.9075128 | 0.1506529 |
| Fixed bilinear base | 31.02918 | 0.9069107 | 0.1521653 |
| Two dense 64-channel convolutions | 31.01534 | 0.9072426 | 0.1515562 |

## Quality: fxaa

| Model | Y PSNR (dB) | Y SSIM | LPIPS (lower is better) |
|---|---:|---:|---:|
| Original 100-epoch reference | 27.49601 | 0.8409153 | 0.1907038 |
| Fixed inputs | 27.49323 | 0.8410279 | 0.1899956 |
| Fixed bilinear base | 27.49550 | 0.8408862 | 0.1913341 |
| Two dense 64-channel convolutions | 27.49479 | 0.8409791 | 0.1902805 |

## Quality: smaa

| Model | Y PSNR (dB) | Y SSIM | LPIPS (lower is better) |
|---|---:|---:|---:|
| Original 100-epoch reference | 27.14702 | 0.8500336 | 0.1718359 |
| Fixed inputs | 27.13668 | 0.8499534 | 0.1709006 |
| Fixed bilinear base | 27.14225 | 0.8498959 | 0.1723199 |
| Two dense 64-channel convolutions | 27.14698 | 0.8499749 | 0.1715372 |

## Quality: native

| Model | Y PSNR (dB) | Y SSIM | LPIPS (lower is better) |
|---|---:|---:|---:|
| Original 100-epoch reference | 26.68192 | 0.8472377 | 0.1618819 |
| Fixed inputs | 26.67595 | 0.8472196 | 0.1606809 |
| Fixed bilinear base | 26.69011 | 0.8473977 | 0.1620965 |
| Two dense 64-channel convolutions | 26.68661 | 0.8473489 | 0.1609100 |

## Paired architecture contrasts

Changed model minus original reference. Positive PSNR/SSIM and negative LPIPS favor the changed model. Nominal intervals and family-wise simultaneous bands are exploratory paired hierarchical bootstrap estimates; 45 metric contrasts share one family. They do not quantify training-seed variation.

| Contrast | Metric | Difference | Nominal 95% interval | Simultaneous 95% band |
|---|---|---:|---|---|
| fixed_inputs minus reference / ssaa_area | y_psnr_shave2_db | +0.00675052 | [-0.00152888, +0.0148177] | [-0.00701233, +0.0205134] |
| fixed_inputs minus reference / ssaa_area | y_ssim_shave2 | +0.000204675 | [+7.22316e-05, +0.00034179] | [-2.17237e-05, +0.000431074] |
| fixed_inputs minus reference / ssaa_area | lpips_alex_v01 | -0.00170445 | [-0.00311858, -0.000432822] | [-0.00397657, +0.000567678] |
| fixed_base minus reference / ssaa_area | y_psnr_shave2_db | -0.0138582 | [-0.0258655, -0.00439993] | [-0.0316137, +0.00389739] |
| fixed_base minus reference / ssaa_area | y_ssim_shave2 | -0.000289721 | [-0.000431089, -0.000164804] | [-0.000498453, -8.0988e-05] |
| fixed_base minus reference / ssaa_area | lpips_alex_v01 | +5.25676e-05 | [-0.000929076, +0.000642815] | [-0.00133147, +0.0014366] |
| dense_core minus reference / ssaa_area | y_psnr_shave2_db | -0.0202368 | [-0.0310298, -0.0110499] | [-0.0370828, -0.00339069] |
| dense_core minus reference / ssaa_area | y_ssim_shave2 | -4.56486e-05 | [-0.000110036, +2.11415e-05] | [-0.000152865, +6.15677e-05] |
| dense_core minus reference / ssaa_area | lpips_alex_v01 | +6.41113e-05 | [-0.000140213, +0.000314679] | [-0.000308564, +0.000436786] |
| fixed_inputs minus reference / bicubic_aa | y_psnr_shave2_db | +0.023671 | [+0.0117933, +0.0357972] | [+0.00356574, +0.0437763] |
| fixed_inputs minus reference / bicubic_aa | y_ssim_shave2 | +0.000438769 | [+0.000223002, +0.000679814] | [+5.72395e-05, +0.000820299] |
| fixed_inputs minus reference / bicubic_aa | lpips_alex_v01 | -0.00132327 | [-0.00207905, -0.000549871] | [-0.00256194, -8.45971e-05] |
| fixed_base minus reference / bicubic_aa | y_psnr_shave2_db | -0.00388202 | [-0.0133146, +0.00379709] | [-0.0178769, +0.0101128] |
| fixed_base minus reference / bicubic_aa | y_ssim_shave2 | -0.000163329 | [-0.000229297, -9.19305e-05] | [-0.000274165, -5.24923e-05] |
| fixed_base minus reference / bicubic_aa | lpips_alex_v01 | +0.000189139 | [-0.000388203, +0.000777533] | [-0.000796607, +0.00117488] |
| dense_core minus reference / bicubic_aa | y_psnr_shave2_db | -0.0177175 | [-0.0347635, -0.00389831] | [-0.0437412, +0.00830614] |
| dense_core minus reference / bicubic_aa | y_ssim_shave2 | +0.000168559 | [+1.28118e-05, +0.000347583] | [-0.000112723, +0.000449842] |
| dense_core minus reference / bicubic_aa | lpips_alex_v01 | -0.000419963 | [-0.00108015, +0.000227227] | [-0.00145014, +0.000610219] |
| fixed_inputs minus reference / fxaa | y_psnr_shave2_db | -0.00277944 | [-0.005855, +0.000717197] | [-0.00814525, +0.00258637] |
| fixed_inputs minus reference / fxaa | y_ssim_shave2 | +0.000112547 | [-6.31766e-05, +0.000338481] | [-0.000224051, +0.000449144] |
| fixed_inputs minus reference / fxaa | lpips_alex_v01 | -0.000708151 | [-0.00135103, -4.5571e-05] | [-0.00174248, +0.000326181] |
| fixed_base minus reference / fxaa | y_psnr_shave2_db | -0.000501977 | [-0.00330973, +0.00192016] | [-0.00487041, +0.00386645] |
| fixed_base minus reference / fxaa | y_ssim_shave2 | -2.91056e-05 | [-7.40909e-05, +1.10871e-05] | [-0.000100218, +4.20068e-05] |
| fixed_base minus reference / fxaa | lpips_alex_v01 | +0.00063035 | [+0.000389269, +0.000964038] | [+0.000138994, +0.00112171] |
| dense_core minus reference / fxaa | y_psnr_shave2_db | -0.00121324 | [-0.00624988, +0.00244305] | [-0.0084491, +0.00602263] |
| dense_core minus reference / fxaa | y_ssim_shave2 | +6.37272e-05 | [-0.000110196, +0.000272906] | [-0.000253819, +0.000381273] |
| dense_core minus reference / fxaa | lpips_alex_v01 | -0.000423342 | [-0.000728308, -0.000134509] | [-0.000913121, +6.64376e-05] |
| fixed_inputs minus reference / smaa | y_psnr_shave2_db | -0.010334 | [-0.0144717, -0.00520531] | [-0.0182182, -0.00244986] |
| fixed_inputs minus reference / smaa | y_ssim_shave2 | -8.027e-05 | [-0.000228001, +7.4124e-05] | [-0.000331496, +0.000170956] |
| fixed_inputs minus reference / smaa | lpips_alex_v01 | -0.000935241 | [-0.00191991, +5.07718e-05] | [-0.00256565, +0.000695166] |
| fixed_base minus reference / smaa | y_psnr_shave2_db | -0.00476851 | [-0.00998148, -2.01177e-05] | [-0.0131048, +0.00356777] |
| fixed_base minus reference / smaa | y_ssim_shave2 | -0.00013773 | [-0.000241802, -3.64834e-05] | [-0.000309999, +3.45385e-05] |
| fixed_base minus reference / smaa | lpips_alex_v01 | +0.000484 | [+1.8391e-05, +0.000841041] | [-0.000170392, +0.00113839] |
| dense_core minus reference / smaa | y_psnr_shave2_db | -3.98562e-05 | [-0.00614342, +0.00655806] | [-0.010513, +0.0104333] |
| dense_core minus reference / smaa | y_ssim_shave2 | -5.8695e-05 | [-0.000177334, +5.1754e-05] | [-0.000250156, +0.000132766] |
| dense_core minus reference / smaa | lpips_alex_v01 | -0.000298659 | [-0.000628502, -1.34127e-05] | [-0.000822021, +0.000224704] |
| fixed_inputs minus reference / native | y_psnr_shave2_db | -0.00597143 | [-0.011541, +0.000487644] | [-0.016003, +0.00406013] |
| fixed_inputs minus reference / native | y_ssim_shave2 | -1.81118e-05 | [-0.000169914, +0.000129737] | [-0.000274142, +0.000237919] |
| fixed_inputs minus reference / native | lpips_alex_v01 | -0.001201 | [-0.00268131, +0.000119447] | [-0.00355878, +0.00115678] |
| fixed_base minus reference / native | y_psnr_shave2_db | +0.00818483 | [+0.000812445, +0.0144708] | [-0.00321932, +0.019589] |
| fixed_base minus reference / native | y_ssim_shave2 | +0.00016001 | [-6.94435e-06, +0.000329322] | [-0.000115819, +0.00043584] |
| fixed_base minus reference / native | lpips_alex_v01 | +0.00021463 | [-0.000491289, +0.000731034] | [-0.000746354, +0.00117561] |
| dense_core minus reference / native | y_psnr_shave2_db | +0.00468662 | [-0.00378826, +0.0132461] | [-0.00967373, +0.019047] |
| dense_core minus reference / native | y_ssim_shave2 | +0.00011121 | [-7.05413e-05, +0.000319356] | [-0.000217251, +0.000439672] |
| dense_core minus reference / native | lpips_alex_v01 | -0.000971929 | [-0.00177694, -0.000272006] | [-0.0022156, +0.000271737] |

## Area FP8 stress test

Area-only uniform controller FP8 E4M3 rounding simulation: per-output-channel weights, per-layer per-tensor activation scales from calibration maxima; FP32 convolution accumulation, biases, analysis and reconstruction; no bias correction; not a native latency or mixed-runtime test

Scales use the same 24 fixed training-crop identities and calibration budget for every model. Each change below compares a model against its own FP16 result. This is not the production stage-specific mixed model, the older calibrated native head stress test, or a latency comparison.

| Model | FP8 Y PSNR | Delta Y PSNR | Delta Y SSIM | Delta LPIPS |
|---|---:|---:|---:|---:|
| Original 100-epoch reference | 31.25929 | -0.03766 | -0.0009532 | -0.0002404 |
| Fixed inputs | 31.25087 | -0.05284 | -0.0009754 | +0.0006182 |
| Fixed bilinear base | 31.20577 | -0.07733 | -0.0015164 | +0.0036226 |
| Two dense 64-channel convolutions | 31.20162 | -0.07510 | -0.0017286 | +0.0018483 |

Precision sensitivity is reported separately from absolute reconstruction quality; the table above is regenerated from these checkpoints rather than retaining earlier rankings.

A separate nine-contrast family compares the within-model quantization changes against the reference. Negative PSNR/SSIM and positive LPIPS differences indicate greater degradation than the reference.

| Control | Metric | Difference in quantization change | Simultaneous 95% band |
|---|---|---:|---|
| fixed_inputs minus reference: quantization change | y_psnr_shave2_db | -0.0151726 | [-0.0320972, +0.0017521] |
| fixed_inputs minus reference: quantization change | y_ssim_shave2 | -0.0000222 | [-0.0003445, +0.0003001] |
| fixed_inputs minus reference: quantization change | lpips_alex_v01 | +0.0008587 | [-0.0008371, +0.0025544] |
| fixed_base minus reference: quantization change | y_psnr_shave2_db | -0.0396673 | [-0.0856030, +0.0062684] |
| fixed_base minus reference: quantization change | y_ssim_shave2 | -0.0005632 | [-0.0015166, +0.0003901] |
| fixed_base minus reference: quantization change | lpips_alex_v01 | +0.0038631 | [+0.0011577, +0.0065685] |
| dense_core minus reference: quantization change | y_psnr_shave2_db | -0.0374365 | [-0.0774775, +0.0026046] |
| dense_core minus reference: quantization change | y_ssim_shave2 | -0.0007754 | [-0.0015077, -0.0000431] |
| dense_core minus reference: quantization change | lpips_alex_v01 | +0.0020887 | [+0.0003552, +0.0038223] |

## Interpretation limits

Small training-loss changes need not predict held-out reconstruction changes. Compare each factor across the five conditions rather than selecting its most favorable track. A single seeded run cannot establish a reproducible optimization advantage. Learned input/base benefit should be described as modest or unresolved where appropriate; their rationale is preserving operator structure, not proving that every fixed alternative has equal optimized cost.

No universal conclusion about parallel bottlenecks follows from this one dense two-layer control. Repeated seeds would measure training-run variation.

Source identities, checkpoint hashes, calibration hashes, folding/reconstruction checks and metric parity checks are recorded in protocol.json and preflight.json. Per-image scores and output hashes are recorded in per_image.jsonl. No held-out image affected calibration or checkpoint selection.

## Output verification

verification.json checks complete unique coverage and the presence of all 28,800 output files. An independent 48-output sample spans the first and last images in every model/precision/condition slot; file hashes, float32 decoding, PNG rounding and recomputed quality metrics pass. This sampled content audit is distinct from checking that every output file exists.
