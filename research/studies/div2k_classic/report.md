# DIV2K-trained AEISU: classic bicubic X2 evaluation

Quality only. Released comparators retain their original training data and objectives. No latency claims.

800 official DIV2K training pairs; 1,000 epochs, fixed stride-16 offsets, D4, MSE, cosine 2e-3 to 1e-6. All 219 test images are evaluation-only. FP16-controller/FP32-reconstruction AEISU; released neural adapters use their recorded precision. No mixed-runtime result is inferred from this checkpoint test.

Primary Y metrics shave two HR pixels; LPIPS-Alex v0.1 uses cropped RGB. B100 is labeled BSD100 in the archive.

## Set5

| Method | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_fp16 | 37.05443 | 0.9570878 | 0.0590124 |
| fsrcnn_x2 | 37.15381 | 0.9569098 | 0.0584632 |
| bilinear | 32.25795 | 0.9120699 | 0.1702821 |
| fsr1_easu | 33.86457 | 0.9313026 | 0.1223284 |
| repnet_x2 | 35.84785 | 0.9450917 | 0.1066712 |
| rt4ksr_x2 | 36.14678 | 0.9519204 | 0.0691513 |
| quicksrnet_small_x2 | 37.19457 | 0.9575556 | 0.0568566 |
| quicksrnet_medium_x2 | 37.43443 | 0.9586577 | 0.0571759 |
| span_x2 | 38.09927 | 0.9610237 | 0.0555635 |
| swinir_light_x2 | 38.18304 | 0.9614009 | 0.0540025 |

## Set14

| Method | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_fp16 | 32.68537 | 0.9105526 | 0.1032750 |
| fsrcnn_x2 | 32.74425 | 0.9097008 | 0.1066792 |
| bilinear | 29.25276 | 0.8408415 | 0.2353207 |
| fsr1_easu | 30.45678 | 0.8676259 | 0.1834449 |
| repnet_x2 | 32.00266 | 0.8942673 | 0.1663404 |
| rt4ksr_x2 | 32.09028 | 0.9038358 | 0.1224117 |
| quicksrnet_small_x2 | 32.83380 | 0.9106428 | 0.0994750 |
| quicksrnet_medium_x2 | 33.05171 | 0.9129664 | 0.0987535 |
| span_x2 | 33.70642 | 0.9191394 | 0.0953724 |
| swinir_light_x2 | 33.87025 | 0.9207911 | 0.0882398 |

## BSD100

| Method | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_fp16 | 31.60849 | 0.8924951 | 0.1526839 |
| fsrcnn_x2 | 31.51424 | 0.8903879 | 0.1608492 |
| bilinear | 28.68795 | 0.8107461 | 0.3045910 |
| fsr1_easu | 29.57181 | 0.8394973 | 0.2473300 |
| repnet_x2 | 30.90689 | 0.8717155 | 0.2250044 |
| rt4ksr_x2 | 31.07024 | 0.8855304 | 0.1850017 |
| quicksrnet_small_x2 | 31.61554 | 0.8924389 | 0.1513260 |
| quicksrnet_medium_x2 | 31.81755 | 0.8954543 | 0.1491387 |
| span_x2 | 32.23541 | 0.9005751 | 0.1474300 |
| swinir_light_x2 | 32.32490 | 0.9015918 | 0.1404729 |

## Urban100

| Method | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_fp16 | 30.30432 | 0.9087473 | 0.0903060 |
| fsrcnn_x2 | 29.84797 | 0.9017365 | 0.1057995 |
| bilinear | 25.96859 | 0.8091720 | 0.2493265 |
| fsr1_easu | 27.20750 | 0.8459439 | 0.1884028 |
| repnet_x2 | 28.80042 | 0.8796739 | 0.1564503 |
| rt4ksr_x2 | 28.74346 | 0.8857038 | 0.1346552 |
| quicksrnet_small_x2 | 30.17861 | 0.9067067 | 0.0942761 |
| quicksrnet_medium_x2 | 30.77319 | 0.9142767 | 0.0824557 |
| span_x2 | 32.21180 | 0.9292399 | 0.0657048 |
| swinir_light_x2 | 32.78484 | 0.9343601 | 0.0570550 |
