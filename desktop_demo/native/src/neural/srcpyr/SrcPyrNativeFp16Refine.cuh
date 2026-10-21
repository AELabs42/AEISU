#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace aess::srcpyr
{
    // Total packed uint4 count: 4 channel groups * 18 steps * 32 threads = 2,304 uint4s (36,864 bytes)
    inline constexpr size_t kNativeFp16RefinePackedU4Count = 4 * 18 * 32;
    inline constexpr size_t kNativeFp16RefinePackedBytes = kNativeFp16RefinePackedU4Count * sizeof(uint4); // 36,864

    // ----------------------------------------------------------------------------
    // Weight Packing for Native SM120 m16n8k16.row.col.f32.f16.f16.f32
    // ----------------------------------------------------------------------------
    // Permutes standard OIHW [64, 32, 3, 3] FP16 weights directly into register-preloaded
    // layout. Each thread holds 18 uint4s across the 18 MMA steps.
    inline void PackNativeFp16RefineWeights(
        const half* w_oihw,
        uint4* outPackedMma)
    {
        auto getWeight = [&](int o, int i, int r, int s) -> half {
            int idx = ((o * 32 + i) * 3 + r) * 3 + s;
            return w_oihw[idx];
        };

        for (int chGrp = 0; chGrp < 4; ++chGrp) {
            int chBase = chGrp * 16;
            int step = 0;
            for (int r = 0; r < 3; ++r) {
                for (int s = 0; s < 3; ++s) {
                    for (int kChunk = 0; kChunk < 2; ++kChunk) {
                        int kBase = kChunk * 16;
                        for (int lane = 0; lane < 32; ++lane) {
                            int n = lane / 4; // 0..7
                            int l = lane % 4; // 0..3

                            // Sub-block A (channels chBase + n)
                            int outA = chBase + n;
                            half aK0 = getWeight(outA, kBase + 4 * l + 0, r, s);
                            half aK1 = getWeight(outA, kBase + 4 * l + 1, r, s);
                            half aK2 = getWeight(outA, kBase + 4 * l + 2, r, s);
                            half aK3 = getWeight(outA, kBase + 4 * l + 3, r, s);

                            uint16_t h0A_0 = *reinterpret_cast<const uint16_t*>(&aK0);
                            uint16_t h0A_1 = *reinterpret_cast<const uint16_t*>(&aK1);
                            uint16_t h0A_2 = *reinterpret_cast<const uint16_t*>(&aK2);
                            uint16_t h0A_3 = *reinterpret_cast<const uint16_t*>(&aK3);

                            uint32_t b0A = (uint32_t)h0A_0 | ((uint32_t)h0A_1 << 16);
                            uint32_t b1A = (uint32_t)h0A_2 | ((uint32_t)h0A_3 << 16);

                            // Sub-block B (channels chBase + 8 + n)
                            int outB = chBase + 8 + n;
                            half bK0 = getWeight(outB, kBase + 4 * l + 0, r, s);
                            half bK1 = getWeight(outB, kBase + 4 * l + 1, r, s);
                            half bK2 = getWeight(outB, kBase + 4 * l + 2, r, s);
                            half bK3 = getWeight(outB, kBase + 4 * l + 3, r, s);

                            uint16_t h0B_0 = *reinterpret_cast<const uint16_t*>(&bK0);
                            uint16_t h0B_1 = *reinterpret_cast<const uint16_t*>(&bK1);
                            uint16_t h0B_2 = *reinterpret_cast<const uint16_t*>(&bK2);
                            uint16_t h0B_3 = *reinterpret_cast<const uint16_t*>(&bK3);

                            uint32_t b0B = (uint32_t)h0B_0 | ((uint32_t)h0B_1 << 16);
                            uint32_t b1B = (uint32_t)h0B_2 | ((uint32_t)h0B_3 << 16);

                            int dstIdx = chGrp * (18 * 32) + step * 32 + lane;
                            outPackedMma[dstIdx] = make_uint4(b0A, b1A, b0B, b1B);
                        }
                        step++;
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Native SM120 FP16 Refine Conv Kernel (16x8 CTA tile)
    // ----------------------------------------------------------------------------
    __global__ void __launch_bounds__(256, 2) native_fp16_refine_conv_kernel(
        uint32_t width,
        uint32_t height,
        const half* __restrict__ input,
        half* __restrict__ output,
        const uint4* __restrict__ weightsMma,
        const float* __restrict__ bias,
        const float* __restrict__ prelu)
    {
        constexpr int C_IN = 32;
        constexpr int C_OUT = 64;
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        constexpr int PATCH_W = TILE_W + 2; // 18
        constexpr int PATCH_H = TILE_H + 2; // 10

        int tileX = blockIdx.x * TILE_W;
        int tileY = blockIdx.y * TILE_H;

        // Patch: 10 * 18 * 32 * sizeof(half) = 11,520 bytes
        // Output staging: 8 * 16 * 64 * sizeof(half) = 16,384 bytes
        __shared__ alignas(16) half s_patch[PATCH_H][PATCH_W][C_IN];
        __shared__ alignas(16) half s_out[TILE_H][TILE_W][C_OUT];

        int tid = threadIdx.y * 32 + threadIdx.x; // 0..255
        int chGroupIdx = threadIdx.y % 4;         // 4 channel groups of 16 = 64
        int chGroup = chGroupIdx * 16;
        int lane = threadIdx.x;                   // 0..31
        int warpRowBase = (threadIdx.y / 4) * 4;  // 0 or 4

        // Preload weights into registers: 18 uint4s per thread (72 registers)
        uint4 regW[18];
        #pragma unroll
        for (int step = 0; step < 18; ++step) {
            regW[step] = weightsMma[chGroupIdx * (18 * 32) + step * 32 + lane];
        }

        int ca0 = chGroup + (lane % 4) * 2;
        int cb0 = chGroup + 8 + (lane % 4) * 2;

        float biasA0 = bias[ca0];
        float biasA1 = bias[ca0 + 1];
        float biasB0 = bias[cb0];
        float biasB1 = bias[cb0 + 1];

        float preluA0 = prelu ? prelu[ca0] : 0.1f;
        float preluA1 = prelu ? prelu[ca0 + 1] : 0.1f;
        float preluB0 = prelu ? prelu[cb0] : 0.1f;
        float preluB1 = prelu ? prelu[cb0 + 1] : 0.1f;

        // Load spatial patch into shared memory: 10x18x32 half = 5,760 halfs = 720 uint4s
        #pragma unroll
        for (int idx = tid; idx < 720; idx += 256) {
            int pPix = idx / 4;
            int cChunk = (idx % 4) * 8;
            int py = pPix / PATCH_W;
            int px = pPix % PATCH_W;
            int gy = tileY + py - 1;
            int gx = tileX + px - 1;

            if (gy >= 0 && gy < static_cast<int>(height) && gx >= 0 && gx < static_cast<int>(width)) {
                *reinterpret_cast<uint4*>(&s_patch[py][px][cChunk]) = 
                    *reinterpret_cast<const uint4*>(input + (gy * width + gx) * C_IN + cChunk);
            } else {
                *reinterpret_cast<uint4*>(&s_patch[py][px][cChunk]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        int l = lane % 4;
        int r0 = lane / 4;   // pixel 0..7
        int r1 = r0 + 8;     // pixel 8..15
        int colOffset = l * 4;

        #pragma unroll
        for (int subRow = 0; subRow < 4; ++subRow) {
            int rowIdx = warpRowBase + subRow;
            if (tileY + rowIdx >= static_cast<int>(height)) continue;

            float d0A = 0, d1A = 0, d2A = 0, d3A = 0;
            float d0B = 0, d1B = 0, d2B = 0, d3B = 0;

            int stepIdx = 0;
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int s = 0; s < 3; ++s) {
                    int inY = rowIdx + r;
                    int inXBase0 = r0 + s;
                    int inXBase1 = r1 + s;

                    #pragma unroll
                    for (int kChunk = 0; kChunk < 2; ++kChunk) {
                        int kBase = kChunk * 16 + colOffset;

                        uint2 v0 = *reinterpret_cast<const uint2*>(&s_patch[inY][inXBase0][kBase]);
                        uint2 v1 = *reinterpret_cast<const uint2*>(&s_patch[inY][inXBase1][kBase]);

                        uint32_t a0 = v0.x;
                        uint32_t a2 = v0.y;
                        uint32_t a1 = v1.x;
                        uint32_t a3 = v1.y;

                        uint4 w = regW[stepIdx];

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,  %1,  %2,  %3},"
                            "{%4,  %5,  %6,  %7},"
                            "{%8,  %9},"
                            "{%10, %11, %12, %13};\n"
                            : "=f"(d0A), "=f"(d1A), "=f"(d2A), "=f"(d3A)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(w.x), "r"(w.y),
                              "f"(d0A), "f"(d1A), "f"(d2A), "f"(d3A)
                        );

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,  %1,  %2,  %3},"
                            "{%4,  %5,  %6,  %7},"
                            "{%8,  %9},"
                            "{%10, %11, %12, %13};\n"
                            : "=f"(d0B), "=f"(d1B), "=f"(d2B), "=f"(d3B)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(w.z), "r"(w.w),
                              "f"(d0B), "f"(d1B), "f"(d2B), "f"(d3B)
                        );

                        stepIdx++;
                    }
                }
            }

            // Fused Bias + strictly 0.1 LeakyReLU in registers!
            auto lrelu = [](float acc, float b, float p) -> half {
                float v = acc + b;
                return __float2half_rn(v > 0.0f ? v : v * p);
            };

            half out0A0 = lrelu(d0A, biasA0, preluA0);
            half out0A1 = lrelu(d1A, biasA1, preluA1);
            half out0B0 = lrelu(d0B, biasB0, preluB0);
            half out0B1 = lrelu(d1B, biasB1, preluB1);

            half out1A0 = lrelu(d2A, biasA0, preluA0);
            half out1A1 = lrelu(d3A, biasA1, preluA1);
            half out1B0 = lrelu(d2B, biasB0, preluB0);
            half out1B1 = lrelu(d3B, biasB1, preluB1);

            // Store to s_out:
            s_out[rowIdx][r0][ca0]     = out0A0;
            s_out[rowIdx][r0][ca0 + 1] = out0A1;
            s_out[rowIdx][r0][cb0]     = out0B0;
            s_out[rowIdx][r0][cb0 + 1] = out0B1;

            s_out[rowIdx][r1][ca0]     = out1A0;
            s_out[rowIdx][r1][ca0 + 1] = out1A1;
            s_out[rowIdx][r1][cb0]     = out1B0;
            s_out[rowIdx][r1][cb0 + 1] = out1B1;
        }
        __syncthreads();

        // Store outputs to global memory in 128-bit coalesced vector stores:
        // 8 rows x 16 cols x 64 channels = 8,192 halfs = 1,024 uint4s
        #pragma unroll
        for (int idx = tid; idx < 1024; idx += 256) {
            int pPix = idx / 8;
            int cChunk = (idx % 8) * 8;
            int py = pPix / TILE_W;
            int px = pPix % TILE_W;
            int gy = tileY + py;
            int gx = tileX + px;

            if (gy < static_cast<int>(height) && gx < static_cast<int>(width)) {
                *reinterpret_cast<uint4*>(output + (gy * width + gx) * C_OUT + cChunk) = 
                    *reinterpret_cast<const uint4*>(&s_out[py][px][cChunk]);
            }
        }
    }

    // Launch helper for Native SM120 FP16 Refine Convolution
    inline void LaunchNativeFp16RefineConv(
        cudaStream_t stream,
        uint32_t width,
        uint32_t height,
        const void* inputFp16,
        void* outputFp16,
        const void* weightsMma,
        const float* bias,
        const float* prelu = nullptr)
    {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        dim3 block(32, 8); // 256 threads
        dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

        native_fp16_refine_conv_kernel<<<grid, block, 0, stream>>>(
            width, height,
            reinterpret_cast<const half*>(inputFp16),
            reinterpret_cast<half*>(outputFp16),
            reinterpret_cast<const uint4*>(weightsMma),
            bias,
            prelu);
    }

    // ----------------------------------------------------------------------------
    // Tail Metadata Head Weight Packing (1x1, 64->64)
    // ----------------------------------------------------------------------------
    // 8 warps * 4 K-chunks * 32 threads = 1,024 uint2s = 512 uint4s (8,192 bytes)
    inline constexpr size_t kNativeFp16TailPackedU4Count = 8 * 2 * 32;
    inline constexpr size_t kNativeFp16TailPackedBytes = kNativeFp16TailPackedU4Count * sizeof(uint4); // 8,192

    inline void PackNativeFp16TailWeights(
        const half* w_tail, // [64, 64] row-major (out, in)
        uint4* outPackedTail // 512 uint4s
    ) {
        for (int warp = 0; warp < 8; ++warp) {
            int cBase = warp * 8;
            for (int lane = 0; lane < 32; ++lane) {
                int n = lane / 4; // 0..7
                int l = lane % 4; // 0..3
                int outCh = cBase + n;

                uint32_t b[4][2];
                for (int kChunk = 0; kChunk < 4; ++kChunk) {
                    int kBase = kChunk * 16 + 4 * l;
                    half k0 = w_tail[outCh * 64 + kBase + 0];
                    half k1 = w_tail[outCh * 64 + kBase + 1];
                    half k2 = w_tail[outCh * 64 + kBase + 2];
                    half k3 = w_tail[outCh * 64 + kBase + 3];

                    uint16_t h0 = *reinterpret_cast<const uint16_t*>(&k0);
                    uint16_t h1 = *reinterpret_cast<const uint16_t*>(&k1);
                    uint16_t h2 = *reinterpret_cast<const uint16_t*>(&k2);
                    uint16_t h3 = *reinterpret_cast<const uint16_t*>(&k3);

                    b[kChunk][0] = (uint32_t)h0 | ((uint32_t)h1 << 16);
                    b[kChunk][1] = (uint32_t)h2 | ((uint32_t)h3 << 16);
                }

                int baseIdx = (warp * 2) * 32 + lane;
                outPackedTail[baseIdx + 0]  = make_uint4(b[0][0], b[0][1], b[1][0], b[1][1]);
                outPackedTail[baseIdx + 32] = make_uint4(b[2][0], b[2][1], b[3][0], b[3][1]);
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Fused Native SM120 FP16 Refine + Tail Convolution Kernel
    // Fuses Refine Conv 3x3 and 1x1 Tail GEMM in registers, eliminating 140 MB of
    // DRAM activation roundtrips.
    // ----------------------------------------------------------------------------
    __global__ void __launch_bounds__(256, 2) fused_native_fp16_refine_tail_conv_kernel(
        uint32_t width,
        uint32_t height,
        const half* __restrict__ input,
        half* __restrict__ outputTail,
        const uint4* __restrict__ refineWeightsMma,
        const float* __restrict__ refineBias,
        const float* __restrict__ refinePrelu,
        const uint4* __restrict__ tailWeightsMma,
        const half* __restrict__ tailBias)
    {
        constexpr int C_IN = 32;
        constexpr int C_REFINE = 64;
        constexpr int C_TAIL = 64;
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        constexpr int PATCH_W = TILE_W + 2;
        constexpr int PATCH_H = TILE_H + 2;

        int tileX = blockIdx.x * TILE_W;
        int tileY = blockIdx.y * TILE_H;

        __shared__ alignas(16) half s_patch[PATCH_H][PATCH_W][C_IN];
        __shared__ alignas(16) half s_out[TILE_H][TILE_W][C_REFINE];
        __shared__ alignas(16) half s_tail[TILE_H][TILE_W][C_TAIL];

        int tid = threadIdx.y * 32 + threadIdx.x;
        int warpId = threadIdx.y; // 0..7
        int chGroupIdx = threadIdx.y % 4;
        int chGroup = chGroupIdx * 16;
        int lane = threadIdx.x;
        int warpRowBase = (threadIdx.y / 4) * 4;

        // 1. Preload Refine Weights into registers
        uint4 regW[18];
        #pragma unroll
        for (int step = 0; step < 18; ++step) {
            regW[step] = refineWeightsMma[chGroupIdx * (18 * 32) + step * 32 + lane];
        }

        int ca0 = chGroup + (lane % 4) * 2;
        int cb0 = chGroup + 8 + (lane % 4) * 2;

        float biasA0 = refineBias[ca0];
        float biasA1 = refineBias[ca0 + 1];
        float biasB0 = refineBias[cb0];
        float biasB1 = refineBias[cb0 + 1];

        float preluA0 = refinePrelu ? refinePrelu[ca0] : 0.1f;
        float preluA1 = refinePrelu ? refinePrelu[ca0 + 1] : 0.1f;
        float preluB0 = refinePrelu ? refinePrelu[cb0] : 0.1f;
        float preluB1 = refinePrelu ? refinePrelu[cb0 + 1] : 0.1f;

        // Load spatial patch into shared memory
        #pragma unroll
        for (int idx = tid; idx < 720; idx += 256) {
            int pPix = idx / 4;
            int cChunk = (idx % 4) * 8;
            int py = pPix / PATCH_W;
            int px = pPix % PATCH_W;
            int gy = tileY + py - 1;
            int gx = tileX + px - 1;

            if (gy >= 0 && gy < static_cast<int>(height) && gx >= 0 && gx < static_cast<int>(width)) {
                *reinterpret_cast<uint4*>(&s_patch[py][px][cChunk]) = 
                    *reinterpret_cast<const uint4*>(input + (gy * width + gx) * C_IN + cChunk);
            } else {
                *reinterpret_cast<uint4*>(&s_patch[py][px][cChunk]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        int l = lane % 4;
        int r0 = lane / 4;
        int r1 = r0 + 8;
        int colOffset = l * 4;

        // 2. Compute Refine 3x3 Convolution -> s_out
        #pragma unroll
        for (int subRow = 0; subRow < 4; ++subRow) {
            int rowIdx = warpRowBase + subRow;
            if (tileY + rowIdx >= static_cast<int>(height)) continue;

            float d0A = 0, d1A = 0, d2A = 0, d3A = 0;
            float d0B = 0, d1B = 0, d2B = 0, d3B = 0;

            int stepIdx = 0;
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int s = 0; s < 3; ++s) {
                    int inY = rowIdx + r;
                    int inXBase0 = r0 + s;
                    int inXBase1 = r1 + s;

                    #pragma unroll
                    for (int kChunk = 0; kChunk < 2; ++kChunk) {
                        int kBase = kChunk * 16 + colOffset;

                        uint2 v0 = *reinterpret_cast<const uint2*>(&s_patch[inY][inXBase0][kBase]);
                        uint2 v1 = *reinterpret_cast<const uint2*>(&s_patch[inY][inXBase1][kBase]);

                        uint32_t a0 = v0.x;
                        uint32_t a2 = v0.y;
                        uint32_t a1 = v1.x;
                        uint32_t a3 = v1.y;

                        uint4 w = regW[stepIdx];

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,  %1,  %2,  %3},"
                            "{%4,  %5,  %6,  %7},"
                            "{%8,  %9},"
                            "{%10, %11, %12, %13};\n"
                            : "=f"(d0A), "=f"(d1A), "=f"(d2A), "=f"(d3A)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(w.x), "r"(w.y),
                              "f"(d0A), "f"(d1A), "f"(d2A), "f"(d3A)
                        );

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,  %1,  %2,  %3},"
                            "{%4,  %5,  %6,  %7},"
                            "{%8,  %9},"
                            "{%10, %11, %12, %13};\n"
                            : "=f"(d0B), "=f"(d1B), "=f"(d2B), "=f"(d3B)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(w.z), "r"(w.w),
                              "f"(d0B), "f"(d1B), "f"(d2B), "f"(d3B)
                        );

                        stepIdx++;
                    }
                }
            }

            auto lrelu = [](float acc, float b, float p) -> half {
                float v = acc + b;
                return __float2half_rn(v > 0.0f ? v : v * p);
            };

            s_out[rowIdx][r0][ca0]     = lrelu(d0A, biasA0, preluA0);
            s_out[rowIdx][r0][ca0 + 1] = lrelu(d1A, biasA1, preluA1);
            s_out[rowIdx][r0][cb0]     = lrelu(d0B, biasB0, preluB0);
            s_out[rowIdx][r0][cb0 + 1] = lrelu(d1B, biasB1, preluB1);

            s_out[rowIdx][r1][ca0]     = lrelu(d2A, biasA0, preluA0);
            s_out[rowIdx][r1][ca0 + 1] = lrelu(d3A, biasA1, preluA1);
            s_out[rowIdx][r1][cb0]     = lrelu(d2B, biasB0, preluB0);
            s_out[rowIdx][r1][cb0 + 1] = lrelu(d3B, biasB1, preluB1);
        }
        __syncthreads();

        // 3. Fused Tail 1x1 GEMM:
        // Warp warpId (0..7) computes 8 output channels: tailBase = warpId * 8
        int tailBase = warpId * 8;
        int tailOutCh0 = tailBase + (lane % 4) * 2;
        int tailOutCh1 = tailOutCh0 + 1;
        float tBias0 = __half2float(tailBias[tailOutCh0]);
        float tBias1 = __half2float(tailBias[tailOutCh1]);

        int tailWeightBaseIdx = (warpId * 2) * 32 + lane;
        uint4 tailW0 = tailWeightsMma[tailWeightBaseIdx + 0];
        uint4 tailW1 = tailWeightsMma[tailWeightBaseIdx + 32];

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (tileY + r >= static_cast<int>(height)) continue;

            float tD0 = 0.0f, tD1 = 0.0f, tD2 = 0.0f, tD3 = 0.0f;

            #pragma unroll
            for (int kChunk = 0; kChunk < 4; ++kChunk) {
                int kBase = kChunk * 16 + colOffset;

                uint2 v0 = *reinterpret_cast<const uint2*>(&s_out[r][r0][kBase]);
                uint2 v1 = *reinterpret_cast<const uint2*>(&s_out[r][r1][kBase]);

                uint32_t a0 = v0.x;
                uint32_t a2 = v0.y;
                uint32_t a1 = v1.x;
                uint32_t a3 = v1.y;

                uint32_t b0 = (kChunk == 0) ? tailW0.x : ((kChunk == 1) ? tailW0.z : ((kChunk == 2) ? tailW1.x : tailW1.z));
                uint32_t b1 = (kChunk == 0) ? tailW0.y : ((kChunk == 1) ? tailW0.w : ((kChunk == 2) ? tailW1.y : tailW1.w));

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13};\n"
                    : "=f"(tD0), "=f"(tD1), "=f"(tD2), "=f"(tD3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(tD0), "f"(tD1), "f"(tD2), "f"(tD3)
                );
            }

            s_tail[r][r0][tailOutCh0] = __float2half_rn(tD0 + tBias0);
            s_tail[r][r0][tailOutCh1] = __float2half_rn(tD1 + tBias1);
            s_tail[r][r1][tailOutCh0] = __float2half_rn(tD2 + tBias0);
            s_tail[r][r1][tailOutCh1] = __float2half_rn(tD3 + tBias1);
        }
        __syncthreads();

        // 4. Coalesced 128-bit vector stores directly to outputTail (tailFp16)
        #pragma unroll
        for (int idx = tid; idx < 1024; idx += 256) {
            int pPix = idx / 8;
            int cChunk = (idx % 8) * 8;
            int py = pPix / TILE_W;
            int px = pPix % TILE_W;
            int gy = tileY + py;
            int gx = tileX + px;

            if (gy < static_cast<int>(height) && gx < static_cast<int>(width)) {
                *reinterpret_cast<uint4*>(outputTail + (gy * width + gx) * C_TAIL + cChunk) = 
                    *reinterpret_cast<const uint4*>(&s_tail[py][px][cChunk]);
            }
        }
    }

    // Launch helper for Fused Native SM120 FP16 Refine + Tail Convolution
    inline void LaunchFusedNativeFp16RefineTailConv(
        cudaStream_t stream,
        uint32_t width,
        uint32_t height,
        const void* inputFp16,
        void* outputTailFp16,
        const void* refineWeightsMma,
        const float* refineBias,
        const float* refinePrelu,
        const void* tailWeightsMma,
        const void* tailBiasFp16)
    {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        dim3 block(32, 8); // 256 threads
        dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

        fused_native_fp16_refine_tail_conv_kernel<<<grid, block, 0, stream>>>(
            width, height,
            reinterpret_cast<const half*>(inputFp16),
            reinterpret_cast<half*>(outputTailFp16),
            reinterpret_cast<const uint4*>(refineWeightsMma),
            refineBias,
            refinePrelu,
            reinterpret_cast<const uint4*>(tailWeightsMma),
            reinterpret_cast<const half*>(tailBiasFp16));
    }
}

