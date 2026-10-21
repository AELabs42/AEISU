# DIV2K L1: native mixed-FP8/FP16 quality

Same final 3000-epoch checkpoint and 219 bicubic benchmark images.
24 DIV2K training inputs calibrate scales and bias correction; six other training inputs check calibration.
No benchmark image enters calibration. Floating scores and competitor scores are reused from verified archives.
Native mixed uses FP16 projection/refinement/head and FP8 internal stages, including native half-precision image resources.

## Set5

| Deployment | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_l1_3000_fp16 | 37.09778 | 0.9574110 | 0.0606767 |
| aeisu_div2k_l1_3000_mixed | 37.09041 | 0.9573662 | 0.0607765 |

## Set14

| Deployment | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_l1_3000_fp16 | 32.72101 | 0.9109847 | 0.1040880 |
| aeisu_div2k_l1_3000_mixed | 32.70896 | 0.9108900 | 0.1044362 |

## BSD100

| Deployment | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_l1_3000_fp16 | 31.63001 | 0.8926932 | 0.1529400 |
| aeisu_div2k_l1_3000_mixed | 31.62086 | 0.8925788 | 0.1538937 |

## Urban100

| Deployment | Y PSNR | Y SSIM | LPIPS |
|---|---:|---:|---:|
| aeisu_div2k_l1_3000_fp16 | 30.36543 | 0.9096901 | 0.0892359 |
| aeisu_div2k_l1_3000_mixed | 30.34623 | 0.9094892 | 0.0897488 |
