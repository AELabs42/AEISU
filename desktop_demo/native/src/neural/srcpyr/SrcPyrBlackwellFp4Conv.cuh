#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cstring>

namespace aess::srcpyr
{
    inline constexpr uint32_t kBlackwellFp4PackedU4Count = 4 * 18 * 32; // 2,304 uint4
    inline constexpr uint32_t kBlackwellFp4PackedBytes = kBlackwellFp4PackedU4Count * sizeof(uint4); // 36,864 bytes

    // Pre-packs 18,432 bytes of packed FP4 nibbles (OIHW format [64, 64, 3, 3])
    // into the 2,304 uint4 (36,864 bytes) warp-register MMA layout for
    // mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32.
    inline void PackBlackwellFp4BottleneckWeights(
        const uint8_t* packedOihwNibbles,
        uint4* outPackedMma)
    {
        auto getNibble = [&](int o, int i, int r, int s) -> uint8_t
        {
            int idx = ((o * 64 + i) * 3 + r) * 3 + s;
            uint8_t byte = packedOihwNibbles[idx >> 1];
            return (idx & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
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
                            uint8_t aK0 = getNibble(outA, cinBase + 2 * l + 0, r, s) << 2;
                            uint8_t aK1 = getNibble(outA, cinBase + 2 * l + 1, r, s) << 2;
                            uint8_t aK8 = getNibble(outA, cinBase + 2 * l + 8, r, s) << 2;
                            uint8_t aK9 = getNibble(outA, cinBase + 2 * l + 9, r, s) << 2;
                            uint32_t b0A = (uint32_t)aK0 | ((uint32_t)aK1 << 8) | ((uint32_t)aK8 << 16) | ((uint32_t)aK9 << 24);

                            uint8_t aK16 = getNibble(outA, cinBase + 2 * l + 16, r, s) << 2;
                            uint8_t aK17 = getNibble(outA, cinBase + 2 * l + 17, r, s) << 2;
                            uint8_t aK24 = getNibble(outA, cinBase + 2 * l + 24, r, s) << 2;
                            uint8_t aK25 = getNibble(outA, cinBase + 2 * l + 25, r, s) << 2;
                            uint32_t b1A = (uint32_t)aK16 | ((uint32_t)aK17 << 8) | ((uint32_t)aK24 << 16) | ((uint32_t)aK25 << 24);

                            // Sub-block B (channels chBase + 8 + n)
                            int outB = chBase + 8 + n;
                            uint8_t bK0 = getNibble(outB, cinBase + 2 * l + 0, r, s) << 2;
                            uint8_t bK1 = getNibble(outB, cinBase + 2 * l + 1, r, s) << 2;
                            uint8_t bK8 = getNibble(outB, cinBase + 2 * l + 8, r, s) << 2;
                            uint8_t bK9 = getNibble(outB, cinBase + 2 * l + 9, r, s) << 2;
                            uint32_t b0B = (uint32_t)bK0 | ((uint32_t)bK1 << 8) | ((uint32_t)bK8 << 16) | ((uint32_t)bK9 << 24);

                            uint8_t bK16 = getNibble(outB, cinBase + 2 * l + 16, r, s) << 2;
                            uint8_t bK17 = getNibble(outB, cinBase + 2 * l + 17, r, s) << 2;
                            uint8_t bK24 = getNibble(outB, cinBase + 2 * l + 24, r, s) << 2;
                            uint8_t bK25 = getNibble(outB, cinBase + 2 * l + 25, r, s) << 2;
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

    __device__ inline int SwizzleChannelOffset(int c)
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

    __global__ void __launch_bounds__(256, 2) blackwell_fp4_bottleneck_conv_kernel(
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

        // Preload weights into registers
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
        float preluA0 = prelu[ca0];
        float preluA1 = prelu[ca0 + 1];

        float scaleB0 = actScale * weightScales[cb0];
        float scaleB1 = actScale * weightScales[cb0 + 1];
        float biasB0 = bias[cb0];
        float biasB1 = bias[cb0 + 1];
        float preluB0 = prelu[cb0];
        float preluB1 = prelu[cb0 + 1];

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
                    int sc = SwizzleChannelOffset(c);
                    *reinterpret_cast<uint16_t*>(&s_patch[py][px][sc]) = inPtr[p];
                }
            }
            else
            {
                #pragma unroll
                for (int p = 0; p < 8; ++p)
                {
                    int c = cChunk + p * 2;
                    int sc = SwizzleChannelOffset(c);
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
                            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
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
                            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
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
                float val = acc * scale + b;
                if (val < 0.0f) val *= pr;
                return __nv_fp8_e4m3(val * outRecipScale).__x;
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

    inline void LaunchBlackwellFp4BottleneckConv(
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
        blackwell_fp4_bottleneck_conv_kernel<<<grid, block, 0, stream>>>(
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
}
