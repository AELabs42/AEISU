#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace aess::srcpyr
{
    // Total packed uint4 count: 8 N-groups * 2 K-chunks * 32 threads = 512 uint4s (8,192 bytes)
    inline constexpr size_t kNativeFp8ExpandPackedU4Count = 8 * 2 * 32;
    inline constexpr size_t kNativeFp8ExpandPackedBytes = kNativeFp8ExpandPackedU4Count * sizeof(uint4); // 8,192

    // ----------------------------------------------------------------------------
    // Weight Packing for Native SM120 1x1 Expand Conv [128, 64, 1, 1]
    // ----------------------------------------------------------------------------
    inline void PackNativeFp8ExpandWeights(
        const uint8_t* oihwWeights, // [128, 64]
        uint4* outPackedMma         // [512 uint4s]
    ) {
        for (int nGrp = 0; nGrp < 8; ++nGrp) {
            int nBase = nGrp * 16;
            for (int kChunk = 0; kChunk < 2; ++kChunk) {
                int kBase = kChunk * 32;
                for (int lane = 0; lane < 32; ++lane) {
                    int n = lane / 4;
                    int l = lane % 4;

                    // Sub-block A (channels nBase + n)
                    int outA = nBase + n;
                    uint8_t aK0 = oihwWeights[outA * 64 + kBase + 2 * l + 0];
                    uint8_t aK1 = oihwWeights[outA * 64 + kBase + 2 * l + 1];
                    uint8_t aK8 = oihwWeights[outA * 64 + kBase + 2 * l + 8];
                    uint8_t aK9 = oihwWeights[outA * 64 + kBase + 2 * l + 9];
                    uint32_t b0A = (uint32_t)aK0 | ((uint32_t)aK1 << 8) | ((uint32_t)aK8 << 16) | ((uint32_t)aK9 << 24);

                    uint8_t aK16 = oihwWeights[outA * 64 + kBase + 2 * l + 16];
                    uint8_t aK17 = oihwWeights[outA * 64 + kBase + 2 * l + 17];
                    uint8_t aK24 = oihwWeights[outA * 64 + kBase + 2 * l + 24];
                    uint8_t aK25 = oihwWeights[outA * 64 + kBase + 2 * l + 25];
                    uint32_t b1A = (uint32_t)aK16 | ((uint32_t)aK17 << 8) | ((uint32_t)aK24 << 16) | ((uint32_t)aK25 << 24);

                    // Sub-block B (channels nBase + 8 + n)
                    int outB = nBase + 8 + n;
                    uint8_t bK0 = oihwWeights[outB * 64 + kBase + 2 * l + 0];
                    uint8_t bK1 = oihwWeights[outB * 64 + kBase + 2 * l + 1];
                    uint8_t bK8 = oihwWeights[outB * 64 + kBase + 2 * l + 8];
                    uint8_t bK9 = oihwWeights[outB * 64 + kBase + 2 * l + 9];
                    uint32_t b0B = (uint32_t)bK0 | ((uint32_t)bK1 << 8) | ((uint32_t)bK8 << 16) | ((uint32_t)bK9 << 24);

                    uint8_t bK16 = oihwWeights[outB * 64 + kBase + 2 * l + 16];
                    uint8_t bK17 = oihwWeights[outB * 64 + kBase + 2 * l + 17];
                    uint8_t bK24 = oihwWeights[outB * 64 + kBase + 2 * l + 24];
                    uint8_t bK25 = oihwWeights[outB * 64 + kBase + 2 * l + 25];
                    uint32_t b1B = (uint32_t)bK16 | ((uint32_t)bK17 << 8) | ((uint32_t)bK24 << 16) | ((uint32_t)bK25 << 24);

                    int dstIdx = (nGrp * 2 + kChunk) * 32 + lane;
                    outPackedMma[dstIdx] = make_uint4(b0A, b1A, b0B, b1B);
                }
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Native Fused SM120 FP8 Expand 1x1 + 0.1 LeakyReLU + PixelShuffle + SkipAdd
    // ----------------------------------------------------------------------------
    // Processes 16 low-res pixels per CTA, expanding to 64 high-res pixels with 32 channels.
    __global__ void __launch_bounds__(256, 4) fused_native_fp8_expand_pixelshuffle_skipadd_kernel(
        uint32_t lowW, uint32_t lowH,
        const uint8_t* __restrict__ inputFp8,
        const half* __restrict__ skipFp16,
        half* __restrict__ outputFp16,
        const uint4* __restrict__ weightsMma,
        float actScale,
        const float* __restrict__ weightScales,
        const float* __restrict__ bias,
        const float* __restrict__ prelu,
        float outRecipScale,
        float dequantScale
    ) {
        int pixBase = blockIdx.x * 16;
        int numPixels = static_cast<int>(lowW * lowH);
        if (pixBase >= numPixels) return;

        int warpId = threadIdx.y; // 0..7 (handles 16 output channels)
        int lane = threadIdx.x;   // 0..31

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

        // Preload weights for this warp: 2 K-chunks
        uint4 regW0 = weightsMma[(warpId * 2 + 0) * 32 + lane];
        uint4 regW1 = weightsMma[(warpId * 2 + 1) * 32 + lane];

        // Shared memory staging:
        // Input: 16 low-res pixels * 64 channels = 1,024 bytes
        // Output FP8 staging: 16 low-res pixels * 4 phases * 32 channels = 2,048 bytes
        __shared__ alignas(16) uint8_t s_in[16][64];
        __shared__ alignas(16) uint8_t s_high_fp8[16][4][32];

        int tid = warpId * 32 + lane; // 0..255

        // Cooperative 32-bit coalesced input load: 256 threads * 4 bytes = 1,024 bytes
        int inP = tid / 16;
        int inC = (tid % 16) * 4;
        int inGlobP = pixBase + inP;
        if (inGlobP < numPixels) {
            *reinterpret_cast<uint32_t*>(&s_in[inP][inC]) = 
                *reinterpret_cast<const uint32_t*>(inputFp8 + size_t(inGlobP) * 64 + inC);
        } else {
            *reinterpret_cast<uint32_t*>(&s_in[inP][inC]) = 0;
        }
        __syncthreads();

        int r0 = lane / 4;      // pixel 0..7
        int r1 = r0 + 8;        // pixel 8..15
        int l = lane % 4;
        int col_base = l * 2;

        const uint8_t* in0 = &s_in[r0][0];
        const uint8_t* in1 = &s_in[r1][0];

        float d0A = 0, d1A = 0, d2A = 0, d3A = 0;
        float d0B = 0, d1B = 0, d2B = 0, d3B = 0;

        // K-chunk 0 (channels 0..31)
        {
            const uint8_t* ptr0 = in0 + col_base;
            const uint8_t* ptr1 = in1 + col_base;

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
                  "r"(regW0.x), "r"(regW0.y),
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
                  "r"(regW0.z), "r"(regW0.w),
                  "f"(d0B), "f"(d1B), "f"(d2B), "f"(d3B)
            );
        }

        // K-chunk 1 (channels 32..63)
        {
            const uint8_t* ptr0 = in0 + 32 + col_base;
            const uint8_t* ptr1 = in1 + 32 + col_base;

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
                  "r"(regW1.x), "r"(regW1.y),
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
                  "r"(regW1.z), "r"(regW1.w),
                  "f"(d0B), "f"(d1B), "f"(d2B), "f"(d3B)
            );
        }

        // Epilogue: strictly 0.1 LeakyReLU + Requantize to FP8 (matching cuDNN output)
        auto epilogueOp = [&](float acc, float scale, float b, float p) -> uint8_t {
            float val = fmaf(acc, scale, b);
            if (val < 0.0f) val *= p; // strictly 0.1 LeakyReLU
            __nv_fp8_e4m3 q(val * outRecipScale);
            return *reinterpret_cast<const uint8_t*>(&q);
        };

        // For pixel r0:
        uint8_t v0A0 = epilogueOp(d0A, scaleA0, biasA0, preluA0);
        uint8_t v0A1 = epilogueOp(d1A, scaleA1, biasA1, preluA1);
        uint8_t v0B0 = epilogueOp(d0B, scaleB0, biasB0, preluB0);
        uint8_t v0B1 = epilogueOp(d1B, scaleB1, biasB1, preluB1);

        // For pixel r1:
        uint8_t v1A0 = epilogueOp(d2A, scaleA0, biasA0, preluA0);
        uint8_t v1A1 = epilogueOp(d3A, scaleA1, biasA1, preluA1);
        uint8_t v1B0 = epilogueOp(d2B, scaleB0, biasB0, preluB0);
        uint8_t v1B1 = epilogueOp(d3B, scaleB1, biasB1, preluB1);

        // Channel mapping to 4 phases: ch = c * 4 + phase
        auto storeToSmem = [&](int pRel, int ch, uint8_t q) {
            int c = ch / 4;
            int phase = ch % 4;
            s_high_fp8[pRel][phase][c] = q;
        };

        storeToSmem(r0, ca0,     v0A0);
        storeToSmem(r0, ca0 + 1, v0A1);
        storeToSmem(r0, cb0,     v0B0);
        storeToSmem(r0, cb0 + 1, v0B1);

        storeToSmem(r1, ca0,     v1A0);
        storeToSmem(r1, ca0 + 1, v1A1);
        storeToSmem(r1, cb0,     v1B0);
        storeToSmem(r1, cb0 + 1, v1B1);

        __syncthreads();

        // Write out to global memory with skip addition:
        // 64 high-res pixels * 32 channels = 2,048 halfs = 256 uint4s (1 uint4 per thread)
        int highPixIdx = tid / 4;     // 0..63
        int cChunk = (tid % 4) * 8;   // 0, 8, 16, 24

        int pLow = highPixIdx / 4;    // 0..15
        int phase = highPixIdx % 4;   // 0..3

        int globalLowPix = pixBase + pLow;
        if (globalLowPix < numPixels) {
            int lx = globalLowPix % static_cast<int>(lowW);
            int ly = globalLowPix / static_cast<int>(lowW);

            int hx = lx * 2 + (phase & 1);
            int hy = ly * 2 + (phase >> 1);

            size_t highOffset = (size_t(hy) * (lowW * 2) + hx) * 32 + cChunk;

            uint4 skipVal4 = *reinterpret_cast<const uint4*>(skipFp16 + highOffset);
            uint2 expVal2 = *reinterpret_cast<const uint2*>(&s_high_fp8[pLow][phase][cChunk]);

            const half* s = reinterpret_cast<const half*>(&skipVal4);
            const uint8_t* qCodes = reinterpret_cast<const uint8_t*>(&expVal2);
            half out[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                __nv_fp8_e4m3 qVal;
                *reinterpret_cast<uint8_t*>(&qVal) = qCodes[i];
                const float combined =
                    float(qVal) * dequantScale +
                    __half2float(s[i]);
                out[i] = __float2half_rn(combined);
            }

            *reinterpret_cast<uint4*>(outputFp16 + highOffset) = *reinterpret_cast<const uint4*>(out);
        }
    }

    // Launch helper for Fused Native SM120 FP8 Expand + PixelShuffle + SkipAdd
    inline void LaunchFusedNativeFp8Expand(
        cudaStream_t stream,
        uint32_t lowW, uint32_t lowH,
        const void* inputFp8,
        const void* skipFp16,
        void* outputFp16,
        const void* weightsMma,
        float actScale,
        const float* weightScales,
        const float* bias,
        const float* prelu,
        float outRecipScale,
        float dequantScale
    ) {
        dim3 block(32, 8); // 256 threads
        dim3 grid((lowW * lowH + 15) / 16);

        fused_native_fp8_expand_pixelshuffle_skipadd_kernel<<<grid, block, 0, stream>>>(
            lowW, lowH,
            reinterpret_cast<const uint8_t*>(inputFp8),
            reinterpret_cast<const half*>(skipFp16),
            reinterpret_cast<half*>(outputFp16),
            reinterpret_cast<const uint4*>(weightsMma),
            actScale, weightScales, bias, prelu,
            outRecipScale, dequantScale);
    }
}
