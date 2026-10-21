#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace aess::srcpyr
{
    // Total packed uint4 count: 4 N-groups * 4 K-chunks * 32 threads = 512 uint4s (8,192 bytes)
    inline constexpr size_t kNativeFp8CompressPackedU4Count = 4 * 4 * 32;
    inline constexpr size_t kNativeFp8CompressPackedBytes = kNativeFp8CompressPackedU4Count * sizeof(uint4); // 8,192

    // ----------------------------------------------------------------------------
    // Weight Packing for Native SM120 1x1 Compress Conv [64, 128, 1, 1]
    // ----------------------------------------------------------------------------
    inline void PackNativeFp8CompressWeights(
        const uint8_t* oihwWeights, // [64, 128]
        uint4* outPackedMma         // [512 uint4s]
    ) {
        for (int nGrp = 0; nGrp < 4; ++nGrp) {
            int nBase = nGrp * 16;
            for (int kChunk = 0; kChunk < 4; ++kChunk) {
                int kBase = kChunk * 32;
                for (int lane = 0; lane < 32; ++lane) {
                    int n = lane / 4;
                    int l = lane % 4;

                    // Sub-block A (channels nBase + n)
                    int outA = nBase + n;
                    uint8_t aK0 = oihwWeights[outA * 128 + kBase + 2 * l + 0];
                    uint8_t aK1 = oihwWeights[outA * 128 + kBase + 2 * l + 1];
                    uint8_t aK8 = oihwWeights[outA * 128 + kBase + 2 * l + 8];
                    uint8_t aK9 = oihwWeights[outA * 128 + kBase + 2 * l + 9];
                    uint32_t b0A = (uint32_t)aK0 | ((uint32_t)aK1 << 8) | ((uint32_t)aK8 << 16) | ((uint32_t)aK9 << 24);

                    uint8_t aK16 = oihwWeights[outA * 128 + kBase + 2 * l + 16];
                    uint8_t aK17 = oihwWeights[outA * 128 + kBase + 2 * l + 17];
                    uint8_t aK24 = oihwWeights[outA * 128 + kBase + 2 * l + 24];
                    uint8_t aK25 = oihwWeights[outA * 128 + kBase + 2 * l + 25];
                    uint32_t b1A = (uint32_t)aK16 | ((uint32_t)aK17 << 8) | ((uint32_t)aK24 << 16) | ((uint32_t)aK25 << 24);

                    // Sub-block B (channels nBase + 8 + n)
                    int outB = nBase + 8 + n;
                    uint8_t bK0 = oihwWeights[outB * 128 + kBase + 2 * l + 0];
                    uint8_t bK1 = oihwWeights[outB * 128 + kBase + 2 * l + 1];
                    uint8_t bK8 = oihwWeights[outB * 128 + kBase + 2 * l + 8];
                    uint8_t bK9 = oihwWeights[outB * 128 + kBase + 2 * l + 9];
                    uint32_t b0B = (uint32_t)bK0 | ((uint32_t)bK1 << 8) | ((uint32_t)bK8 << 16) | ((uint32_t)bK9 << 24);

                    uint8_t bK16 = oihwWeights[outB * 128 + kBase + 2 * l + 16];
                    uint8_t bK17 = oihwWeights[outB * 128 + kBase + 2 * l + 17];
                    uint8_t bK24 = oihwWeights[outB * 128 + kBase + 2 * l + 24];
                    uint8_t bK25 = oihwWeights[outB * 128 + kBase + 2 * l + 25];
                    uint32_t b1B = (uint32_t)bK16 | ((uint32_t)bK17 << 8) | ((uint32_t)bK24 << 16) | ((uint32_t)bK25 << 24);

                    int dstIdx = (nGrp * 4 + kChunk) * 32 + lane;
                    outPackedMma[dstIdx] = make_uint4(b0A, b1A, b0B, b1B);
                }
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Fused Native SM120 FP8 Compress Conv:
    // Unshuffles skipFp16 (32 channels, 2x2 high-res) -> FP8 -> 1x1 Conv (128->64)
    // -> 0.1 LeakyReLU -> Requantize to FP8 -> writes directly to lowAFp8
    // ----------------------------------------------------------------------------
    __global__ void __launch_bounds__(128, 8) fused_native_fp8_compress_kernel(
        uint32_t lowW, uint32_t lowH,
        const half* __restrict__ skipFp16,    // [highH * highW, 32]
        uint8_t* __restrict__ outputLowFp8,   // [lowH * lowW, 64]
        const uint4* __restrict__ weightsMma, // [512 uint4s]
        float recipInScale,                   // 1.0f / activationScales[1]
        float actScale,                       // activationScales[1]
        const float* __restrict__ weightScales,
        const float* __restrict__ bias,
        const float* __restrict__ prelu,
        float outRecipScale                   // 1.0f / activationScales[2]
    ) {
        int pixBase = blockIdx.x * 16;
        int numPixels = static_cast<int>(lowW * lowH);
        if (pixBase >= numPixels) return;

        int warpId = threadIdx.y; // 0..3 (handles 16 output channels: [16*warpId .. 16*warpId+15])
        int lane = threadIdx.x;   // 0..31
        int tid = warpId * 32 + lane; // 0..127

        // Shared memory: 16 low-res pixels * 128 channels = 2,048 bytes
        __shared__ alignas(16) uint8_t s_in[16][128];
        // Output staging: 16 low-res pixels * 64 channels = 1,024 bytes
        __shared__ alignas(16) uint8_t s_out[16][64];

        // Pass 0: Row 0 (phases 0 and 1) - 1 uint4 per thread
        {
            int pLow = tid / 8;           // 0..15
            int chunk = tid % 8;          // 0..7
            int phase = chunk / 4;        // 0 or 1
            int cChunk = (chunk % 4) * 8; // 0, 8, 16, 24
            int p = pixBase + pLow;
            if (p < numPixels) {
                int lx = p % static_cast<int>(lowW);
                int ly = p / static_cast<int>(lowW);
                int hx = lx * 2 + phase;
                int hy = ly * 2 + 0;
                size_t highOffset = (size_t(hy) * (lowW * 2) + hx) * 32 + cChunk;
                uint4 val4 = *reinterpret_cast<const uint4*>(skipFp16 + highOffset);
                const half* h = reinterpret_cast<const half*>(&val4);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int c = cChunk + i;
                    int k = c * 4 + phase;
                    __nv_fp8_e4m3 q(__half2float(h[i]) * recipInScale);
                    s_in[pLow][k] = *reinterpret_cast<const uint8_t*>(&q);
                }
            }
        }

        // Pass 1: Row 1 (phases 2 and 3) - 1 uint4 per thread
        {
            int pLow = tid / 8;           // 0..15
            int chunk = tid % 8;          // 0..7
            int phase = 2 + (chunk / 4);  // 2 or 3
            int cChunk = (chunk % 4) * 8; // 0, 8, 16, 24
            int p = pixBase + pLow;
            if (p < numPixels) {
                int lx = p % static_cast<int>(lowW);
                int ly = p / static_cast<int>(lowW);
                int hx = lx * 2 + (phase & 1);
                int hy = ly * 2 + 1;
                size_t highOffset = (size_t(hy) * (lowW * 2) + hx) * 32 + cChunk;
                uint4 val4 = *reinterpret_cast<const uint4*>(skipFp16 + highOffset);
                const half* h = reinterpret_cast<const half*>(&val4);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int c = cChunk + i;
                    int k = c * 4 + phase;
                    __nv_fp8_e4m3 q(__half2float(h[i]) * recipInScale);
                    s_in[pLow][k] = *reinterpret_cast<const uint8_t*>(&q);
                }
            }
        }
        __syncthreads();

        // Preload weights for this warp: 4 K-chunks
        uint4 regW0 = weightsMma[(warpId * 4 + 0) * 32 + lane];
        uint4 regW1 = weightsMma[(warpId * 4 + 1) * 32 + lane];
        uint4 regW2 = weightsMma[(warpId * 4 + 2) * 32 + lane];
        uint4 regW3 = weightsMma[(warpId * 4 + 3) * 32 + lane];

        int chGroup = warpId * 16;
        int ca0 = chGroup + (lane % 4) * 2;
        int cb0 = chGroup + 8 + (lane % 4) * 2;

        float scaleA0 = actScale * weightScales[ca0];
        float scaleA1 = actScale * weightScales[ca0 + 1];
        float biasA0 = bias[ca0];
        float biasA1 = bias[ca0 + 1];

        float scaleB0 = actScale * weightScales[cb0];
        float scaleB1 = actScale * weightScales[cb0 + 1];
        float biasB0 = bias[cb0];
        float biasB1 = bias[cb0 + 1];

        float preluA0 = prelu ? prelu[ca0] : 0.1f;
        float preluA1 = prelu ? prelu[ca0 + 1] : 0.1f;
        float preluB0 = prelu ? prelu[cb0] : 0.1f;
        float preluB1 = prelu ? prelu[cb0 + 1] : 0.1f;

        int r0 = lane / 4;      // pixel 0..7
        int r1 = r0 + 8;        // pixel 8..15
        int l = lane % 4;
        int col_base = l * 2;

        const uint8_t* in0 = &s_in[r0][0];
        const uint8_t* in1 = &s_in[r1][0];

        float d0A = 0, d1A = 0, d2A = 0, d3A = 0;
        float d0B = 0, d1B = 0, d2B = 0, d3B = 0;

        auto mmaChunk = [&](int kBase, uint4 regW) {
            const uint8_t* ptr0 = in0 + kBase + col_base;
            const uint8_t* ptr1 = in1 + kBase + col_base;

            uint16_t p0_0  = *reinterpret_cast<const uint16_t*>(ptr0);
            uint16_t p0_8  = *reinterpret_cast<const uint16_t*>(ptr0 + 8);
            uint16_t p0_16 = *reinterpret_cast<const uint16_t*>(ptr0 + 16);
            uint16_t p0_24 = *reinterpret_cast<const uint16_t*>(ptr0 + 24);

            uint16_t p1_0  = *reinterpret_cast<const uint16_t*>(ptr1);
            uint16_t p1_8  = *reinterpret_cast<const uint16_t*>(ptr1 + 8);
            uint16_t p1_16 = *reinterpret_cast<const uint16_t*>(ptr1 + 16);
            uint16_t p1_24 = *reinterpret_cast<const uint16_t*>(ptr1 + 24);

            uint32_t a0 = (uint32_t)p0_0  | ((uint32_t)p0_8 << 16);
            uint32_t a2 = (uint32_t)p0_16 | ((uint32_t)p0_24 << 16);
            uint32_t a1 = (uint32_t)p1_0  | ((uint32_t)p1_8 << 16);
            uint32_t a3 = (uint32_t)p1_16 | ((uint32_t)p1_24 << 16);

            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(d0A), "=f"(d1A), "=f"(d2A), "=f"(d3A)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(regW.x), "r"(regW.y),
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
                  "r"(regW.z), "r"(regW.w),
                  "f"(d0B), "f"(d1B), "f"(d2B), "f"(d3B)
            );
        };

        mmaChunk(0,  regW0);
        mmaChunk(32, regW1);
        mmaChunk(64, regW2);
        mmaChunk(96, regW3);

        // Epilogue: strictly 0.1 LeakyReLU + Requantize to FP8 (matching cuDNN output)
        auto epilogueOp = [&](float acc, float scale, float b, float p) -> uint8_t {
            float val = fmaf(acc, scale, b);
            if (val < 0.0f) val *= p; // strictly 0.1 LeakyReLU
            __nv_fp8_e4m3 q(val * outRecipScale);
            return *reinterpret_cast<const uint8_t*>(&q);
        };

        uint8_t v0A0 = epilogueOp(d0A, scaleA0, biasA0, preluA0);
        uint8_t v0A1 = epilogueOp(d1A, scaleA1, biasA1, preluA1);
        uint8_t v0B0 = epilogueOp(d0B, scaleB0, biasB0, preluB0);
        uint8_t v0B1 = epilogueOp(d1B, scaleB1, biasB1, preluB1);

        uint8_t v1A0 = epilogueOp(d2A, scaleA0, biasA0, preluA0);
        uint8_t v1A1 = epilogueOp(d3A, scaleA1, biasA1, preluA1);
        uint8_t v1B0 = epilogueOp(d2B, scaleB0, biasB0, preluB0);
        uint8_t v1B1 = epilogueOp(d3B, scaleB1, biasB1, preluB1);

        // Store to shared memory staging:
        s_out[r0][ca0]     = v0A0;
        s_out[r0][ca0 + 1] = v0A1;
        s_out[r0][cb0]     = v0B0;
        s_out[r0][cb0 + 1] = v0B1;

        s_out[r1][ca0]     = v1A0;
        s_out[r1][ca0 + 1] = v1A1;
        s_out[r1][cb0]     = v1B0;
        s_out[r1][cb0 + 1] = v1B1;

        __syncthreads();

        // Vectorized 128-bit store to global memory:
        // 16 low-res pixels * 64 channels = 1024 bytes = 64 uint4s
        // Exactly 64 of the 128 threads write 1 uint4 (16 bytes)
        if (tid < 64) {
            int p = tid / 4;         // 0..15
            int cBase = (tid % 4) * 16; // 0, 16, 32, 48
            int globP = pixBase + p;
            if (globP < numPixels) {
                uint4 outVal = *reinterpret_cast<const uint4*>(&s_out[p][cBase]);
                *reinterpret_cast<uint4*>(outputLowFp8 + size_t(globP) * 64 + cBase) = outVal;
            }
        }
    }

    // Launch helper for Fused Native SM120 FP8 Compress Conv
    inline void LaunchFusedNativeFp8Compress(
        cudaStream_t stream,
        uint32_t lowW, uint32_t lowH,
        const void* skipFp16,
        void* outputLowFp8,
        const void* weightsMma,
        float recipInScale,
        float actScale,
        const float* weightScales,
        const float* bias,
        const float* prelu,
        float outRecipScale
    ) {
        dim3 block(32, 4); // 128 threads
        dim3 grid((lowW * lowH + 15) / 16);

        fused_native_fp8_compress_kernel<<<grid, block, 0, stream>>>(
            lowW, lowH,
            reinterpret_cast<const half*>(skipFp16),
            reinterpret_cast<uint8_t*>(outputLowFp8),
            reinterpret_cast<const uint4*>(weightsMma),
            recipInScale, actScale,
            weightScales, bias, prelu,
            outRecipScale);
    }
}
