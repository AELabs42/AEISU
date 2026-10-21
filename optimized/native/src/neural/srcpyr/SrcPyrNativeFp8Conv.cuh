#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstring>

namespace aess::srcpyr
{
    inline constexpr uint32_t kNativeFp8PackedU4Count = 4 * 18 * 32; // 2,304 uint4
    inline constexpr uint32_t kNativeFp8PackedBytes = kNativeFp8PackedU4Count * sizeof(uint4); // 36,864 bytes

    inline constexpr uint32_t kNativeFp8Grouped64PackedU4Count = 4 * 9 * 32; // 1,152 uint4
    inline constexpr uint32_t kNativeFp8Grouped64PackedBytes = kNativeFp8Grouped64PackedU4Count * sizeof(uint4); // 18,432 bytes

    inline constexpr uint32_t kNativeFp16Grouped64PackedHalfs = 64 * 32 * 9; // 18,432 halfs
    inline constexpr uint32_t kNativeFp16Grouped64PackedBytes = kNativeFp16Grouped64PackedHalfs * sizeof(half); // 36,864 bytes

    // Pre-packs 36,864 bytes of FP16 weights (OIHW format [64, 32, 3, 3] groups=2)
    // into 16x16 col-major tiles for WMMA Tensor Core execution.
    inline void PackNativeFp16GroupedWeights(
        const half* oihwWeights,
        half* outPackedMma)
    {
        for (int g = 0; g < 2; ++g) {
            for (int tap = 0; tap < 9; ++tap) {
                for (int n = 0; n < 2; ++n) {
                    for (int k = 0; k < 2; ++k) {
                        half* tilePtr = &outPackedMma[(((g * 9 + tap) * 2 + n) * 2 + k) * 256];
                        for (int n_sub = 0; n_sub < 16; ++n_sub) {
                            for (int k_sub = 0; k_sub < 16; ++k_sub) {
                                int o = g * 32 + n * 16 + n_sub;
                                int i = k * 16 + k_sub;
                                half val = oihwWeights[(o * 32 + i) * 9 + tap];
                                // col-major tile: tilePtr[n_sub * 16 + k_sub]
                                tilePtr[n_sub * 16 + k_sub] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    // Pre-packs 18,432 bytes of FP8 weights (OIHW format [64, 32, 3, 3] groups=2)
    // into the 1,152 uint4 (18,432 bytes) warp-register MMA layout for
    // mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.
    inline void PackNativeFp8GroupedWeights(
        const uint8_t* oihwWeights,
        uint4* outPackedMma)
    {
        auto getWeight = [&](int o, int i, int r, int s) -> uint8_t
        {
            int idx = ((o * 32 + i) * 3 + r) * 3 + s;
            return oihwWeights[idx];
        };

        for (int chGrp = 0; chGrp < 4; ++chGrp)
        {
            int chBase = chGrp * 16;
            int step = 0;
            for (int r = 0; r < 3; ++r)
            {
                for (int s = 0; s < 3; ++s)
                {
                    for (int lane = 0; lane < 32; ++lane)
                    {
                        int n = lane / 4;
                        int l = lane % 4;

                        // Sub-block A (channels chBase + n)
                        int outA = chBase + n;
                        uint8_t aK0 = getWeight(outA, 2 * l + 0, r, s);
                        uint8_t aK1 = getWeight(outA, 2 * l + 1, r, s);
                        uint8_t aK8 = getWeight(outA, 2 * l + 8, r, s);
                        uint8_t aK9 = getWeight(outA, 2 * l + 9, r, s);
                        uint32_t b0A = (uint32_t)aK0 | ((uint32_t)aK1 << 8) | ((uint32_t)aK8 << 16) | ((uint32_t)aK9 << 24);

                        uint8_t aK16 = getWeight(outA, 2 * l + 16, r, s);
                        uint8_t aK17 = getWeight(outA, 2 * l + 17, r, s);
                        uint8_t aK24 = getWeight(outA, 2 * l + 24, r, s);
                        uint8_t aK25 = getWeight(outA, 2 * l + 25, r, s);
                        uint32_t b1A = (uint32_t)aK16 | ((uint32_t)aK17 << 8) | ((uint32_t)aK24 << 16) | ((uint32_t)aK25 << 24);

                        // Sub-block B (channels chBase + 8 + n)
                        int outB = chBase + 8 + n;
                        uint8_t bK0 = getWeight(outB, 2 * l + 0, r, s);
                        uint8_t bK1 = getWeight(outB, 2 * l + 1, r, s);
                        uint8_t bK8 = getWeight(outB, 2 * l + 8, r, s);
                        uint8_t bK9 = getWeight(outB, 2 * l + 9, r, s);
                        uint32_t b0B = (uint32_t)bK0 | ((uint32_t)bK1 << 8) | ((uint32_t)bK8 << 16) | ((uint32_t)bK9 << 24);

                        uint8_t bK16 = getWeight(outB, 2 * l + 16, r, s);
                        uint8_t bK17 = getWeight(outB, 2 * l + 17, r, s);
                        uint8_t bK24 = getWeight(outB, 2 * l + 24, r, s);
                        uint8_t bK25 = getWeight(outB, 2 * l + 25, r, s);
                        uint32_t b1B = (uint32_t)bK16 | ((uint32_t)bK17 << 8) | ((uint32_t)bK24 << 16) | ((uint32_t)bK25 << 24);

                        int dstIdx = (chGrp * 9 + step) * 32 + lane;
                        outPackedMma[dstIdx] = make_uint4(b0A, b1A, b0B, b1B);
                    }
                    step++;
                }
            }
        }
    }


    // Pre-packs 36,864 bytes of FP8 weights (OIHW format [64, 64, 3, 3])
    // into the 2,304 uint4 (36,864 bytes) warp-register MMA layout for
    // mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.
    inline void PackNativeFp8BottleneckWeights(
        const uint8_t* oihwWeights,
        uint4* outPackedMma)
    {
        auto getWeight = [&](int o, int i, int r, int s) -> uint8_t
        {
            int idx = ((o * 64 + i) * 3 + r) * 3 + s;
            return oihwWeights[idx];
        };

        for (int chGrp = 0; chGrp < 4; ++chGrp)
        {
            int chBase = chGrp * 16;
            int step = 0;
            for (int r = 0; r < 3; ++r)
            {
                for (int s = 0; s < 3; ++s)
                {
                    for (int cChunk = 0; cChunk < 2; ++cChunk)
                    {
                        int cinBase = cChunk * 32;

                        for (int lane = 0; lane < 32; ++lane)
                        {
                            int n = lane / 4;
                            int l = lane % 4;

                            // Sub-block A (channels chBase + n)
                            int outA = chBase + n;
                            uint8_t aK0 = getWeight(outA, cinBase + 2 * l + 0, r, s);
                            uint8_t aK1 = getWeight(outA, cinBase + 2 * l + 1, r, s);
                            uint8_t aK8 = getWeight(outA, cinBase + 2 * l + 8, r, s);
                            uint8_t aK9 = getWeight(outA, cinBase + 2 * l + 9, r, s);
                            uint32_t b0A = (uint32_t)aK0 | ((uint32_t)aK1 << 8) | ((uint32_t)aK8 << 16) | ((uint32_t)aK9 << 24);

                            uint8_t aK16 = getWeight(outA, cinBase + 2 * l + 16, r, s);
                            uint8_t aK17 = getWeight(outA, cinBase + 2 * l + 17, r, s);
                            uint8_t aK24 = getWeight(outA, cinBase + 2 * l + 24, r, s);
                            uint8_t aK25 = getWeight(outA, cinBase + 2 * l + 25, r, s);
                            uint32_t b1A = (uint32_t)aK16 | ((uint32_t)aK17 << 8) | ((uint32_t)aK24 << 16) | ((uint32_t)aK25 << 24);

                            // Sub-block B (channels chBase + 8 + n)
                            int outB = chBase + 8 + n;
                            uint8_t bK0 = getWeight(outB, cinBase + 2 * l + 0, r, s);
                            uint8_t bK1 = getWeight(outB, cinBase + 2 * l + 1, r, s);
                            uint8_t bK8 = getWeight(outB, cinBase + 2 * l + 8, r, s);
                            uint8_t bK9 = getWeight(outB, cinBase + 2 * l + 9, r, s);
                            uint32_t b0B = (uint32_t)bK0 | ((uint32_t)bK1 << 8) | ((uint32_t)bK8 << 16) | ((uint32_t)bK9 << 24);

                            uint8_t bK16 = getWeight(outB, cinBase + 2 * l + 16, r, s);
                            uint8_t bK17 = getWeight(outB, cinBase + 2 * l + 17, r, s);
                            uint8_t bK24 = getWeight(outB, cinBase + 2 * l + 24, r, s);
                            uint8_t bK25 = getWeight(outB, cinBase + 2 * l + 25, r, s);
                            uint32_t b1B = (uint32_t)bK16 | ((uint32_t)bK17 << 8) | ((uint32_t)bK24 << 16) | ((uint32_t)bK25 << 24);

                            int dstIdx = (chGrp * 18 + step) * 32 + lane;
                            outPackedMma[dstIdx] = make_uint4(b0A, b1A, b0B, b1B);
                        }
                        step++;
                    }
                }
            }
        }
    }

    __device__ inline int SwizzleChannelOffsetFp8(int c)
    {
        int chunk = (c / 32) * 32;
        int rem = c % 32;
        int pair = rem / 2;
        int byteInPair = rem % 2;
        int l = pair % 4;
        int k = pair / 4;
        int swizzledPair = l * 4 + k;
        return chunk + swizzledPair * 2 + byteInPair;
    }

    __global__ void __launch_bounds__(256, 2) native_fp8_bottleneck_conv_kernel(
        uint32_t width,
        uint32_t height,
        const uint8_t* __restrict__ inputFp8,
        uint8_t* __restrict__ outputFp8,
        const uint4* __restrict__ weightsMma,
        float actScale,
        const float* __restrict__ weightScales,
        const float* __restrict__ bias,
        const float* __restrict__ prelu,
        float outRecipScale)
    {
        constexpr int C_IN = 64;
        constexpr int C_OUT = 64;
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 4;
        constexpr int PATCH_W = TILE_W + 2; // 18
        constexpr int PATCH_H = TILE_H + 2; // 6

        int tileX = blockIdx.x * TILE_W;
        int tileY = blockIdx.y * TILE_H;

        __shared__ alignas(16) uint8_t s_patch[PATCH_H][PATCH_W][C_IN];
        __shared__ alignas(16) uint8_t s_out[TILE_H][TILE_W][C_OUT];

        int tid = threadIdx.y * 32 + threadIdx.x; // 0..255

        int chGroupIdx = threadIdx.y % 4;
        int chGroup = chGroupIdx * 16;
        int lane = threadIdx.x;
        int warpRowBase = (threadIdx.y / 4) * 2;

        // Preload weights into registers (18 uint4 = 72 registers)
        uint4 regW[18];
        #pragma unroll
        for (int step = 0; step < 18; ++step)
        {
            regW[step] = weightsMma[chGroupIdx * (18 * 32) + step * 32 + lane];
        }

        int ca0 = chGroup + (lane % 4) * 2;
        int cb0 = chGroup + 8 + (lane % 4) * 2;

        float scaleA0 = actScale * weightScales[ca0];
        float scaleA1 = actScale * weightScales[ca0 + 1];
        float biasA0 = bias[ca0];
        float biasA1 = bias[ca0 + 1];
        float preluA0 = prelu ? prelu[ca0] : 0.1f;
        float preluA1 = prelu ? prelu[ca0 + 1] : 0.1f;

        float scaleB0 = actScale * weightScales[cb0];
        float scaleB1 = actScale * weightScales[cb0 + 1];
        float biasB0 = bias[cb0];
        float biasB1 = bias[cb0 + 1];
        float preluB0 = prelu ? prelu[cb0] : 0.1f;
        float preluB1 = prelu ? prelu[cb0 + 1] : 0.1f;

        #pragma unroll
        for (int idx = tid; idx < 432; idx += 256)
        {
            int pPix = idx / 4;
            int cChunk = (idx % 4) * 16;
            int py = pPix / PATCH_W;
            int px = pPix % PATCH_W;
            int gy = tileY + py - 1;
            int gx = tileX + px - 1;

            if (gy >= 0 && gy < static_cast<int>(height) && gx >= 0 && gx < static_cast<int>(width))
            {
                const uint16_t* inPtr = reinterpret_cast<const uint16_t*>(
                    inputFp8 + (gy * width + gx) * C_IN + cChunk
                );
                #pragma unroll
                for (int p = 0; p < 8; ++p)
                {
                    int c = cChunk + p * 2;
                    int sc = SwizzleChannelOffsetFp8(c);
                    *reinterpret_cast<uint16_t*>(&s_patch[py][px][sc]) = inPtr[p];
                }
            }
            else
            {
                #pragma unroll
                for (int p = 0; p < 8; ++p)
                {
                    int c = cChunk + p * 2;
                    int sc = SwizzleChannelOffsetFp8(c);
                    *reinterpret_cast<uint16_t*>(&s_patch[py][px][sc]) = 0;
                }
            }
        }
        __syncthreads();

        int l = lane % 4;
        int r0 = lane / 4;
        int r1 = r0 + 8;
        int swizzledColOffset = l * 8;

        #pragma unroll
        for (int subRow = 0; subRow < 2; ++subRow)
        {
            int rowIdx = warpRowBase + subRow;

            float d0A = 0, d1A = 0, d2A = 0, d3A = 0;
            float d0B = 0, d1B = 0, d2B = 0, d3B = 0;

            int stepIdx = 0;
            #pragma unroll
            for (int r = 0; r < 3; ++r)
            {
                #pragma unroll
                for (int s = 0; s < 3; ++s)
                {
                    int inY = rowIdx + r;
                    int inXBase0 = r0 + s;
                    int inXBase1 = r1 + s;

                    #pragma unroll
                    for (int cChunk = 0; cChunk < 2; ++cChunk)
                    {
                        int chunkBase = cChunk * 32 + swizzledColOffset;

                        uint2 v0 = *reinterpret_cast<const uint2*>(&s_patch[inY][inXBase0][chunkBase]);
                        uint2 v1 = *reinterpret_cast<const uint2*>(&s_patch[inY][inXBase1][chunkBase]);

                        uint32_t a0 = v0.x;
                        uint32_t a2 = v0.y;
                        uint32_t a1 = v1.x;
                        uint32_t a3 = v1.y;

                        uint4 w = regW[stepIdx];

                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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

            auto calcQ = [&](float acc, float scale, float b, float pr) -> uint8_t
            {
                float val = fmaf(acc, scale, b);
                if (val < 0.0f) val *= pr;
                __nv_fp8_e4m3 q(val * outRecipScale);
                return *reinterpret_cast<const uint8_t*>(&q);
            };

            s_out[rowIdx][r0][ca0]     = calcQ(d0A, scaleA0, biasA0, preluA0);
            s_out[rowIdx][r0][ca0 + 1] = calcQ(d1A, scaleA1, biasA1, preluA1);
            s_out[rowIdx][r1][ca0]     = calcQ(d2A, scaleA0, biasA0, preluA0);
            s_out[rowIdx][r1][ca0 + 1] = calcQ(d3A, scaleA1, biasA1, preluA1);

            s_out[rowIdx][r0][cb0]     = calcQ(d0B, scaleB0, biasB0, preluB0);
            s_out[rowIdx][r0][cb0 + 1] = calcQ(d1B, scaleB1, biasB1, preluB1);
            s_out[rowIdx][r1][cb0]     = calcQ(d2B, scaleB0, biasB0, preluB0);
            s_out[rowIdx][r1][cb0 + 1] = calcQ(d3B, scaleB1, biasB1, preluB1);
        }

        __syncthreads();

        int outPix = tid / 4;
        int cVec = (tid % 4) * 16;
        int py = outPix / 16;
        int px = outPix % 16;
        int gy = tileY + py;
        int gx = tileX + px;

        if (gy < static_cast<int>(height) && gx < static_cast<int>(width))
        {
            uint4 outVal = *reinterpret_cast<const uint4*>(&s_out[py][px][cVec]);
            *reinterpret_cast<uint4*>(outputFp8 + (gy * width + gx) * C_OUT + cVec) = outVal;
        }
    }

    inline void LaunchNativeFp8BottleneckConv(
        cudaStream_t stream,
        uint32_t width,
        uint32_t height,
        const void* inputFp8,
        void* outputFp8,
        const void* packedWeightsMma,
        float actScale,
        const float* weightScales,
        const float* bias,
        const float* prelu,
        float outRecipScale)
    {
        dim3 block(32, 8);
        dim3 grid((width + 15) / 16, (height + 3) / 4);
        native_fp8_bottleneck_conv_kernel<<<grid, block, 0, stream>>>(
            width, height,
            reinterpret_cast<const uint8_t*>(inputFp8),
            reinterpret_cast<uint8_t*>(outputFp8),
            reinterpret_cast<const uint4*>(packedWeightsMma),
            actScale,
            weightScales,
            bias,
            prelu,
            outRecipScale);
    }

    __global__ void __launch_bounds__(256, 2) native_fp8_grouped64_conv_kernel(
        uint32_t width,
        uint32_t height,
        const uint8_t* __restrict__ inputFp8,
        uint8_t* __restrict__ outputFp8,
        const uint4* __restrict__ weightsMmaGrp,
        float actScale,
        const float* __restrict__ weightScales,
        const float* __restrict__ bias,
        const float* __restrict__ prelu,
        float outRecipScale)
    {
        constexpr int C_IN = 64;
        constexpr int C_OUT = 64;
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 4;
        constexpr int PATCH_W = TILE_W + 2; // 18
        constexpr int PATCH_H = TILE_H + 2; // 6

        int tileX = blockIdx.x * TILE_W;
        int tileY = blockIdx.y * TILE_H;

        constexpr int SMEM_C = 80;
        __shared__ alignas(16) uint8_t s_patch[PATCH_H][PATCH_W][SMEM_C];
        __shared__ alignas(16) uint8_t s_out[TILE_H][TILE_W][C_OUT];

        int tid = threadIdx.y * 32 + threadIdx.x; // 0..255

        int chGroupIdx = threadIdx.y % 4;
        int chGroup = chGroupIdx * 16;
        int lane = threadIdx.x;
        int warpRowBase = (threadIdx.y / 4) * 2;
        int inChunkBase = (chGroupIdx / 2) * 32;

        // Preload weights into registers (9 uint4 = 36 registers)
        uint4 regW[9];
        #pragma unroll
        for (int step = 0; step < 9; ++step)
        {
            regW[step] = weightsMmaGrp[chGroupIdx * (9 * 32) + step * 32 + lane];
        }

        int ca0 = chGroup + (lane % 4) * 2;
        int cb0 = chGroup + 8 + (lane % 4) * 2;

        float scaleA0 = actScale * weightScales[ca0];
        float scaleA1 = actScale * weightScales[ca0 + 1];
        float biasA0 = bias[ca0];
        float biasA1 = bias[ca0 + 1];
        float preluA0 = prelu ? prelu[ca0] : 0.1f;
        float preluA1 = prelu ? prelu[ca0 + 1] : 0.1f;

        float scaleB0 = actScale * weightScales[cb0];
        float scaleB1 = actScale * weightScales[cb0 + 1];
        float biasB0 = bias[cb0];
        float biasB1 = bias[cb0 + 1];
        float preluB0 = prelu ? prelu[cb0] : 0.1f;
        float preluB1 = prelu ? prelu[cb0 + 1] : 0.1f;

        #pragma unroll
        for (int idx = tid; idx < 432; idx += 256)
        {
            int pPix = idx / 4;
            int cChunk = (idx % 4) * 16;
            int py = pPix / PATCH_W;
            int px = pPix % PATCH_W;
            int gy = tileY + py - 1;
            int gx = tileX + px - 1;

            uint4 inVal = make_uint4(0, 0, 0, 0);
            if (gy >= 0 && gy < static_cast<int>(height) && gx >= 0 && gx < static_cast<int>(width))
            {
                inVal = *reinterpret_cast<const uint4*>(
                    inputFp8 + (gy * width + gx) * C_IN + cChunk
                );
            }

            uint32_t w0 = (inVal.x & 0x0000FFFFu) | ((inVal.z & 0x0000FFFFu) << 16);
            uint32_t w1 = (inVal.x >> 16)         | ((inVal.z >> 16) << 16);
            uint32_t w2 = (inVal.y & 0x0000FFFFu) | ((inVal.w & 0x0000FFFFu) << 16);
            uint32_t w3 = (inVal.y >> 16)         | ((inVal.w >> 16) << 16);

            int cBase = (cChunk / 32) * 32 + ((cChunk & 16) ? 4 : 0);
            *reinterpret_cast<uint32_t*>(&s_patch[py][px][cBase + 0])  = w0;
            *reinterpret_cast<uint32_t*>(&s_patch[py][px][cBase + 8])  = w1;
            *reinterpret_cast<uint32_t*>(&s_patch[py][px][cBase + 16]) = w2;
            *reinterpret_cast<uint32_t*>(&s_patch[py][px][cBase + 24]) = w3;
        }
        __syncthreads();

        int l = lane % 4;
        int r0 = lane / 4;
        int r1 = r0 + 8;
        int swizzledColOffset = inChunkBase + l * 8;

        int row0 = warpRowBase;
        int row1 = warpRowBase + 1;

        float d0A0 = 0, d1A0 = 0, d2A0 = 0, d3A0 = 0;
        float d0B0 = 0, d1B0 = 0, d2B0 = 0, d3B0 = 0;

        float d0A1 = 0, d1A1 = 0, d2A1 = 0, d3A1 = 0;
        float d0B1 = 0, d1B1 = 0, d2B1 = 0, d3B1 = 0;

        int stepIdx = 0;
        #pragma unroll
        for (int r = 0; r < 3; ++r)
        {
            int inY0 = row0 + r;
            int inY1 = row1 + r;
            #pragma unroll
            for (int s = 0; s < 3; ++s)
            {
                int inXBase0 = r0 + s;
                int inXBase1 = r1 + s;

                uint2 v0_r0 = *reinterpret_cast<const uint2*>(&s_patch[inY0][inXBase0][swizzledColOffset]);
                uint2 v1_r0 = *reinterpret_cast<const uint2*>(&s_patch[inY0][inXBase1][swizzledColOffset]);

                uint2 v0_r1 = *reinterpret_cast<const uint2*>(&s_patch[inY1][inXBase0][swizzledColOffset]);
                uint2 v1_r1 = *reinterpret_cast<const uint2*>(&s_patch[inY1][inXBase1][swizzledColOffset]);

                uint4 w = regW[stepIdx];

                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13};\n"
                    : "=f"(d0A0), "=f"(d1A0), "=f"(d2A0), "=f"(d3A0)
                    : "r"(v0_r0.x), "r"(v1_r0.x), "r"(v0_r0.y), "r"(v1_r0.y),
                      "r"(w.x), "r"(w.y),
                      "f"(d0A0), "f"(d1A0), "f"(d2A0), "f"(d3A0)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13};\n"
                    : "=f"(d0A1), "=f"(d1A1), "=f"(d2A1), "=f"(d3A1)
                    : "r"(v0_r1.x), "r"(v1_r1.x), "r"(v0_r1.y), "r"(v1_r1.y),
                      "r"(w.x), "r"(w.y),
                      "f"(d0A1), "f"(d1A1), "f"(d2A1), "f"(d3A1)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13};\n"
                    : "=f"(d0B0), "=f"(d1B0), "=f"(d2B0), "=f"(d3B0)
                    : "r"(v0_r0.x), "r"(v1_r0.x), "r"(v0_r0.y), "r"(v1_r0.y),
                      "r"(w.z), "r"(w.w),
                      "f"(d0B0), "f"(d1B0), "f"(d2B0), "f"(d3B0)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13};\n"
                    : "=f"(d0B1), "=f"(d1B1), "=f"(d2B1), "=f"(d3B1)
                    : "r"(v0_r1.x), "r"(v1_r1.x), "r"(v0_r1.y), "r"(v1_r1.y),
                      "r"(w.z), "r"(w.w),
                      "f"(d0B1), "f"(d1B1), "f"(d2B1), "f"(d3B1)
                );

                stepIdx++;
            }
        }

        auto calcQ = [&](float acc, float scale, float b, float pr) -> uint8_t
        {
            float val = fmaf(acc, scale, b);
            if (val < 0.0f) val *= pr;
            __nv_fp8_e4m3 q(val * outRecipScale);
            return *reinterpret_cast<const uint8_t*>(&q);
        };

        s_out[row0][r0][ca0]     = calcQ(d0A0, scaleA0, biasA0, preluA0);
        s_out[row0][r0][ca0 + 1] = calcQ(d1A0, scaleA1, biasA1, preluA1);
        s_out[row0][r1][ca0]     = calcQ(d2A0, scaleA0, biasA0, preluA0);
        s_out[row0][r1][ca0 + 1] = calcQ(d3A0, scaleA1, biasA1, preluA1);

        s_out[row0][r0][cb0]     = calcQ(d0B0, scaleB0, biasB0, preluB0);
        s_out[row0][r0][cb0 + 1] = calcQ(d1B0, scaleB1, biasB1, preluB1);
        s_out[row0][r1][cb0]     = calcQ(d2B0, scaleB0, biasB0, preluB0);
        s_out[row0][r1][cb0 + 1] = calcQ(d3B0, scaleB1, biasB1, preluB1);

        s_out[row1][r0][ca0]     = calcQ(d0A1, scaleA0, biasA0, preluA0);
        s_out[row1][r0][ca0 + 1] = calcQ(d1A1, scaleA1, biasA1, preluA1);
        s_out[row1][r1][ca0]     = calcQ(d2A1, scaleA0, biasA0, preluA0);
        s_out[row1][r1][ca0 + 1] = calcQ(d3A1, scaleA1, biasA1, preluA1);

        s_out[row1][r0][cb0]     = calcQ(d0B1, scaleB0, biasB0, preluB0);
        s_out[row1][r0][cb0 + 1] = calcQ(d1B1, scaleB1, biasB1, preluB1);
        s_out[row1][r1][cb0]     = calcQ(d2B1, scaleB0, biasB0, preluB0);
        s_out[row1][r1][cb0 + 1] = calcQ(d3B1, scaleB1, biasB1, preluB1);

        __syncthreads();

        int outPix = tid / 4;
        int cVec = (tid % 4) * 16;
        int py = outPix / 16;
        int px = outPix % 16;
        int gy = tileY + py;
        int gx = tileX + px;

        if (gy < static_cast<int>(height) && gx < static_cast<int>(width))
        {
            uint4 outVal = *reinterpret_cast<const uint4*>(&s_out[py][px][cVec]);
            *reinterpret_cast<uint4*>(outputFp8 + (gy * width + gx) * C_OUT + cVec) = outVal;
        }
    }

    inline void LaunchNativeFp8Grouped64Conv(
        cudaStream_t stream,
        uint32_t width,
        uint32_t height,
        const void* inputFp8,
        void* outputFp8,
        const void* packedWeightsMma,
        float actScale,
        const float* weightScales,
        const float* bias,
        const float* prelu,
        float outRecipScale)
    {
        dim3 block(32, 8);
        dim3 grid((width + 15) / 16, (height + 3) / 4);
        native_fp8_grouped64_conv_kernel<<<grid, block, 0, stream>>>(
            width, height,
            reinterpret_cast<const uint8_t*>(inputFp8),
            reinterpret_cast<uint8_t*>(outputFp8),
            reinterpret_cast<const uint4*>(packedWeightsMma),
            actScale,
            weightScales,
            bias,
            prelu,
            outRecipScale);
    }

    __global__ void __launch_bounds__(256) wmma_fp16_grouped64_conv_kernel(
        uint32_t width,
        uint32_t height,
        const half* __restrict__ input,
        half* __restrict__ output,
        const half* __restrict__ weights, // [64, 9, 32] col-major tiles
        const float* __restrict__ bias,
        const float* __restrict__ prelu)
    {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        constexpr int PATCH_W = TILE_W + 2; // 18
        constexpr int PATCH_H = TILE_H + 2; // 10

        int tileX = blockIdx.x * TILE_W;
        int tileY = blockIdx.y * TILE_H;

        constexpr int SMEM_C = 64;
        union SmemStorage {
            alignas(16) half s_patch[PATCH_H][PATCH_W][SMEM_C];
            alignas(16) float s_warp_out[8][16][32];
        };
        __shared__ alignas(16) SmemStorage smem;
        __shared__ float s_bias[64];
        __shared__ float s_prelu[64];

        int tid = threadIdx.y * 32 + threadIdx.x; // 0..255

        if (tid < 64) {
            s_bias[tid] = bias[tid];
            s_prelu[tid] = prelu ? prelu[tid] : 0.1f;
        }

        #pragma unroll
        for (int idx = tid; idx < 1440; idx += 256) {
            int pPix = idx / 8;
            int cChunk = (idx % 8) * 8;
            int py = pPix / PATCH_W;
            int px = pPix % PATCH_W;
            int gy = tileY + py - 1;
            int gx = tileX + px - 1;

            if (gy >= 0 && gy < static_cast<int>(height) && gx >= 0 && gx < static_cast<int>(width)) {
                *reinterpret_cast<uint4*>(&smem.s_patch[py][px][cChunk]) = 
                    *reinterpret_cast<const uint4*>(input + (gy * width + gx) * 64 + cChunk);
            } else {
                *reinterpret_cast<uint4*>(&smem.s_patch[py][px][cChunk]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        int warpId = tid / 32;
        int lane = tid % 32;

        int group = warpId / 4;           // Group 0 or 1
        int rowInTile = (warpId % 4) * 2; // row 0, 2, 4, 6

        int inChBase = group * 32;
        int outChBase = group * 32;

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag_A[2];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag_B[2];

        nvcuda::wmma::fill_fragment(c_frag_A[0], 0.0f);
        nvcuda::wmma::fill_fragment(c_frag_A[1], 0.0f);
        nvcuda::wmma::fill_fragment(c_frag_B[0], 0.0f);
        nvcuda::wmma::fill_fragment(c_frag_B[1], 0.0f);

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag_A;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag_B;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag[9][2][2];

        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const half* wPtr = weights + (((group * 9 + t) * 2 + n) * 2 + k) * 256;
                    nvcuda::wmma::load_matrix_sync(b_frag[t][n][k], wPtr, 16);
                }
            }
        }

        int tap = 0;
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            #pragma unroll
            for (int s = 0; s < 3; ++s) {
                int inYA = rowInTile + r;
                int inYB = rowInTile + 1 + r;
                int inX = s;

                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    int kOffset = inChBase + k * 16;
                    nvcuda::wmma::load_matrix_sync(a_frag_A, &smem.s_patch[inYA][inX][kOffset], SMEM_C);
                    nvcuda::wmma::load_matrix_sync(a_frag_B, &smem.s_patch[inYB][inX][kOffset], SMEM_C);

                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        nvcuda::wmma::mma_sync(c_frag_A[n], a_frag_A, b_frag[tap][n][k], c_frag_A[n]);
                        nvcuda::wmma::mma_sync(c_frag_B[n], a_frag_B, b_frag[tap][n][k], c_frag_B[n]);
                    }
                }
                tap++;
            }
        }

        __syncthreads();

        // Process row A:
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            nvcuda::wmma::store_matrix_sync(&smem.s_warp_out[warpId][0][n * 16], c_frag_A[n], 32, nvcuda::wmma::mem_row_major);
        }
        __syncwarp();

        int gy_A = tileY + rowInTile;
        if (gy_A < static_cast<int>(height)) {
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                int idx = lane * 16 + e;
                int px = idx / 32;
                int c = idx % 32;
                int gx = tileX + px;
                if (gx < static_cast<int>(width)) {
                    float acc = smem.s_warp_out[warpId][px][c];
                    int globalC = outChBase + c;
                    acc += s_bias[globalC];
                    if (acc < 0.0f) acc *= s_prelu[globalC];
                    output[(gy_A * width + gx) * 64 + globalC] = __float2half(acc);
                }
            }
        }

        // Process row B:
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            nvcuda::wmma::store_matrix_sync(&smem.s_warp_out[warpId][0][n * 16], c_frag_B[n], 32, nvcuda::wmma::mem_row_major);
        }
        __syncwarp();

        int gy_B = tileY + rowInTile + 1;
        if (gy_B < static_cast<int>(height)) {
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                int idx = lane * 16 + e;
                int px = idx / 32;
                int c = idx % 32;
                int gx = tileX + px;
                if (gx < static_cast<int>(width)) {
                    float acc = smem.s_warp_out[warpId][px][c];
                    int globalC = outChBase + c;
                    acc += s_bias[globalC];
                    if (acc < 0.0f) acc *= s_prelu[globalC];
                    output[(gy_B * width + gx) * 64 + globalC] = __float2half(acc);
                }
            }
        }
    }

    inline void LaunchNativeFp16Grouped64Conv(
        cudaStream_t stream,
        uint32_t width,
        uint32_t height,
        const void* inputFp16,
        void* outputFp16,
        const void* packedWeightsMma,
        const float* bias,
        const float* prelu)
    {
        dim3 block(32, 8);
        dim3 grid((width + 15) / 16, (height + 7) / 8);
        wmma_fp16_grouped64_conv_kernel<<<grid, block, 0, stream>>>(
            width, height,
            reinterpret_cast<const half*>(inputFp16),
            reinterpret_cast<half*>(outputFp16),
            reinterpret_cast<const half*>(packedWeightsMma),
            bias,
            prelu);
    }
}
