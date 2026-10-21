#pragma once

// Direct RGB head ablations. Fuse PixelShuffle(4), optional learned base,
// clamp and RGBA16F surface write; no full-resolution RGB intermediate.
namespace aess::adaptive
{
    template <bool AddBase, int BX = 16, int BY = 8>
    __global__ void ReconstructDirectRgb(cudaSurfaceObject_t source,
        const __half* __restrict__ packedRgb, LearnedFilterBank bank,
        cudaSurfaceObject_t output, int width, int height, int packedWidth,
        int padX, int padY)
    {
        constexpr int TW = BX + 4, TH = BY + 4;
        __shared__ float tile[3][TH][TW];
        const int tx = int(threadIdx.x), ty = int(threadIdx.y);
        const int x = int(blockIdx.x) * BX + tx, y = int(blockIdx.y) * BY + ty;
        if constexpr (AddBase)
        {
            for (int i = ty * BX + tx; i < TW * TH; i += BX * BY)
            {
                const int ix = i % TW, iy = i / TW;
                const int sx = min(width - 1, max(0, int(blockIdx.x) * BX + ix - 2));
                const int sy = min(height - 1, max(0, int(blockIdx.y) * BY + iy - 2));
                const ushort4 value = surf2Dread<ushort4>(source, sx * int(sizeof(ushort4)), sy);
                tile[0][iy][ix] = __half2float(__ushort_as_half(value.x));
                tile[1][iy][ix] = __half2float(__ushort_as_half(value.y));
                tile[2][iy][ix] = __half2float(__ushort_as_half(value.z));
            }
            __syncthreads();
        }
        if (x >= width || y >= height) return;
        const int px = x + padX, py = y + padY;
        const size_t offset = (size_t(py / 2) * packedWidth + px / 2) * 64;
        float rgb[3][4];
        #pragma unroll
        for (int color = 0; color < 3; ++color)
        {
            #pragma unroll
            for (int phase = 0; phase < 4; ++phase)
            {
                const int subpixel = ((py & 1) * 2 + (phase >> 1)) * 4 +
                    (px & 1) * 2 + (phase & 1);
                rgb[color][phase] = __half2float(packedRgb[offset + color * 16 + subpixel]);
            }
        }
        if constexpr (AddBase)
        {
            // Match training: compute the base first, then add the residual.
            float base[3][4] = {};
            #pragma unroll
            for (int tap = 0; tap < 25; ++tap)
            {
                const float4 weights = bank.base[tap];
                const float phases[4] = {weights.x, weights.y, weights.z, weights.w};
                #pragma unroll
                for (int color = 0; color < 3; ++color)
                {
                    const float value = tile[color][ty + tap / 5][tx + tap % 5];
                    #pragma unroll
                    for (int phase = 0; phase < 4; ++phase)
                        base[color][phase] = fmaf(value, phases[phase], base[color][phase]);
                }
            }
            #pragma unroll
            for (int color = 0; color < 3; ++color)
                #pragma unroll
                for (int phase = 0; phase < 4; ++phase)
                    rgb[color][phase] += base[color][phase];
        }
        #pragma unroll
        for (int phase = 0; phase < 4; ++phase)
        {
            const ushort4 result = make_ushort4(
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, rgb[0][phase])))),
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, rgb[1][phase])))),
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, rgb[2][phase])))), 0x3c00);
            surf2Dwrite(result, output, (2 * x + (phase & 1)) * int(sizeof(ushort4)), 2 * y + (phase >> 1));
        }
    }
}
