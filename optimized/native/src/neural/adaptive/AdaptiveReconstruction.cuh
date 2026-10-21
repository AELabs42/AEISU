#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>

namespace aess::adaptive
{
    struct BasisBank { float4 values[25 * 16]; };
    struct LearnedFilterBank { float4 values[25 * 16]; float4 base[25]; };
    struct Half4 { __half2 xy; __half2 zw; };
    struct LearnedFilterBankHalf { Half4 values[25 * 16]; Half4 base[25]; };
    struct LearnedPairMoEFilterBank { float4 values[25 * 32]; float4 base[25]; };

    __device__ __forceinline__ __half TransferCoefficientHalf(__half raw)
    {
        const __half magnitude = __habs(raw);
        const __half two = __float2half(2.0f);
        const __half val = magnitude <= two ? magnitude :
            __hmin(__float2half(2.5f), __hfma(__float2half(0.05f), magnitude - two, two));
        return raw < __float2half(0.0f) ? -val : val;
    }

    __device__ __forceinline__ __half2 MakeCoeff2(__half h)
    {
        return __half2half2(TransferCoefficientHalf(h));
    }

    template <bool Overclock, bool Feature8>
    __device__ __forceinline__ float TransferCoefficient(float raw)
    {
        if constexpr (Overclock)
        {
            const float magnitude=fabsf(raw);
            const float value=magnitude<=2.0f ? magnitude :
                fminf(2.5f,fmaf(0.05f,magnitude-2.0f,2.0f));
            return copysignf(value,raw);
        }
        else
        {
            constexpr float limit=Feature8 ? 2.0f : 1.0f;
            return fminf(limit,fmaxf(-limit,raw));
        }
    }
    inline constexpr int kBasisColumns = 112;
    inline constexpr int kSplitBasisElements = 2 * 16 * kBasisColumns;

    // Preserve the FP32 learned basis as a high/low pair, rather than silently
    // replacing it with a single rounded FP16 tensor. Executed once per engine.
    __global__ void PackTensorCoreBasis(const float4* basis, __half* packed)
    {
        const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
        if (index >= kSplitBasisElements) return;
        const int offset = index % (16 * kBasisColumns);
        const int k = offset / kBasisColumns, column = offset % kBasisColumns;
        float value = 0.0f;
        if (column < 100)
        {
            const float4 phases = basis[(column / 4) * 16 + k];
            value = (column & 3) == 0 ? phases.x : ((column & 3) == 1 ? phases.y :
                ((column & 3) == 2 ? phases.z : phases.w));
        }
        const __half high = __float2half_rn(value);
        packed[index] = index < 16 * kBasisColumns ? high :
            __float2half_rn(value - __half2float(high));
    }

    template <int TW, int TH>
    __device__ __forceinline__ float BilinearPhase(
        const float (&tile)[3][TH][TW], int c, int x, int y, int phase)
    {
        const int px = phase & 1, py = phase >> 1;
        const int left = x + 1 + px, top = y + 1 + py;
        const float wx = px ? 0.75f : 0.25f, wy = py ? 0.75f : 0.25f;
        return wy * (wx * tile[c][top][left] + (1.0f - wx) * tile[c][top][left + 1]) +
            (1.0f - wy) * (wx * tile[c][top + 1][left] + (1.0f - wx) * tile[c][top + 1][left + 1]);
    }

    template <int TW, int TH>
    __device__ __forceinline__ float BilinearPhase(
        const ushort4 (&tile)[TH][TW], int c, int x, int y, int phase)
    {
        const int px = phase & 1, py = phase >> 1;
        const int left = x + 1 + px, top = y + 1 + py;
        const float wx = px ? 0.75f : 0.25f, wy = py ? 0.75f : 0.25f;
        auto get_c = [](const ushort4& p, int ch) -> float {
            unsigned short h = (ch == 0) ? p.x : ((ch == 1) ? p.y : p.z);
            return __half2float(__ushort_as_half(h));
        };
        return wy * (wx * get_c(tile[top][left], c) + (1.0f - wx) * get_c(tile[top][left + 1], c)) +
            (1.0f - wy) * (wx * get_c(tile[top + 1][left], c) + (1.0f - wx) * get_c(tile[top + 1][left + 1], c));
    }

    // Each warp mixes sixteen source pixels against all 100 phase/tap weights.
    // Two FP16 MMA products retain the FP32 basis' high and low components.
    // The mixed weights never leave shared memory; RGB reconstruction is fused.
    template <int Warps = 8, int WarpColumns = 2, bool Feature8 = false>
    __global__ void ReconstructTensorCore(cudaSurfaceObject_t source,
        const __half* __restrict__ metadata, const __half* __restrict__ basis,
        cudaSurfaceObject_t output, int width, int height, int packedWidth, int padX, int padY)
    {
        static_assert(Warps % WarpColumns == 0);
        constexpr int BX = WarpColumns * 16, BY = Warps / WarpColumns;
        constexpr int TW = BX + 4, TH = BY + 4, NT = Warps * 32;
        constexpr float maxCoeff = Feature8 ? 2.0f : 1.0f;
        __shared__ float tile[3][TH][TW];
        __shared__ __align__(32) __half filters[2][16 * kBasisColumns];
        __shared__ __align__(32) __half coefficients[Warps][16 * 16];
        __shared__ __align__(32) float weights[Warps][16 * 16];
        const int tid = int(threadIdx.x), warp = tid / 32, lane = tid & 31;
        const int bx = int(blockIdx.x) * BX, by = int(blockIdx.y) * BY;
        const int warpX = (warp % WarpColumns) * 16, warpY = warp / WarpColumns;
        const bool interior = bx >= 2 && by >= 2 && bx + BX + 2 <= width && by + BY + 2 <= height;
        for (int q = tid; q < TW * TH; q += NT)
        {
            const int ty = q / TW, tx = q % TW;
            int sx = bx + tx - 2, sy = by + ty - 2;
            if (!interior) { sx = max(0, min(width - 1, sx)); sy = max(0, min(height - 1, sy)); }
            const ushort4 value = surf2Dread<ushort4>(source, sx * int(sizeof(ushort4)), sy);
            tile[0][ty][tx] = __half2float(__ushort_as_half(value.x));
            tile[1][ty][tx] = __half2float(__ushort_as_half(value.y));
            tile[2][ty][tx] = __half2float(__ushort_as_half(value.z));
        }
        for (int q = tid; q < kSplitBasisElements; q += NT)
            filters[q / (16 * kBasisColumns)][q % (16 * kBasisColumns)] = basis[q];
        #pragma unroll
        for (int q = lane; q < 16 * 16; q += 32)
        {
            const int x = bx + warpX + q / 16, y = by + warpY;
            float value = 0.0f;
            if (x < width && y < height)
            {
                const int px = x + padX, py = y + padY;
                const int parity = ((py & 1) << 1) | (px & 1);
                value = __half2float(metadata[(size_t(py >> 1) * packedWidth + (px >> 1)) * 64 + parity * 16 + q % 16]);
            }
            coefficients[warp][q] = __float2half_rn(fminf(maxCoeff, fmaxf(-maxCoeff, value)));
        }
        __syncthreads();
        const int pixel = lane / 2, phase = lane & 1;
        const int tx = warpX + pixel, ty = warpY;
        const int x = bx + tx, y = by + ty;
        float r0 = BilinearPhase(tile, 0, tx, ty, phase), r1 = BilinearPhase(tile, 0, tx, ty, phase + 2);
        float g0 = BilinearPhase(tile, 1, tx, ty, phase), g1 = BilinearPhase(tile, 1, tx, ty, phase + 2);
        float b0 = BilinearPhase(tile, 2, tx, ty, phase), b1 = BilinearPhase(tile, 2, tx, ty, phase + 2);
        const float cr = tile[0][ty + 2][tx + 2], cg = tile[1][ty + 2][tx + 2], cb = tile[2][ty + 2][tx + 2];
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a;
        nvcuda::wmma::load_matrix_sync(a, coefficients[warp], 16);
        #pragma unroll 1
        for (int column = 0; column < 112; column += 16)
        {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;
            nvcuda::wmma::fill_fragment(c, 0.0f);
            // Accumulate the low product before the high product to retain the
            // small correction as far as FP32 accumulation permits.
            nvcuda::wmma::load_matrix_sync(b, filters[1] + column, kBasisColumns);
            nvcuda::wmma::mma_sync(c, a, b, c);
            nvcuda::wmma::load_matrix_sync(b, filters[0] + column, kBasisColumns);
            nvcuda::wmma::mma_sync(c, a, b, c);
            nvcuda::wmma::store_matrix_sync(weights[warp], c, 16, nvcuda::wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int tap = column / 4 + j;
                if (tap < 25)
                {
                    const int sy = ty + tap / 5, sx = tx + tap % 5;
                    // Zero-DC filters act on differences, guaranteeing exact
                    // flat-color preservation even with split-basis rounding.
                    const float sr = tile[0][sy][sx] - cr, sg = tile[1][sy][sx] - cg, sb = tile[2][sy][sx] - cb;
                    const float w0 = weights[warp][pixel * 16 + j * 4 + phase];
                    const float w1 = weights[warp][pixel * 16 + j * 4 + phase + 2];
                    r0 = fmaf(w0, sr, r0); r1 = fmaf(w1, sr, r1);
                    g0 = fmaf(w0, sg, g0); g1 = fmaf(w1, sg, g1);
                    b0 = fmaf(w0, sb, b0); b1 = fmaf(w1, sb, b1);
                }
            }
            __syncwarp();
        }
        if (x < width && y < height)
        {
            const ushort4 top = make_ushort4(
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, r0)))),
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, g0)))),
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, b0)))), 0x3c00);
            const ushort4 bottom = make_ushort4(
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, r1)))),
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, g1)))),
                __half_as_ushort(__float2half_rn(fminf(1.0f, fmaxf(0.0f, b1)))), 0x3c00);
            surf2Dwrite(top, output, (2 * x + phase) * int(sizeof(ushort4)), 2 * y);
            surf2Dwrite(bottom, output, (2 * x + phase) * int(sizeof(ushort4)), 2 * y + 1);
        }
    }

    // One source pixel produces four output phases. The immutable learned
    // basis belongs to the engine, never a process-global mutable constant.
    // A small block-local copy gives warp broadcasts without lifetime hazards.
    template <int BX=16,int BY=8,bool Feature8=false>
    __global__ void Reconstruct(cudaSurfaceObject_t source,const __half* __restrict__ metadata,
        const float4* __restrict__ basis,cudaSurfaceObject_t output,
        int width,int height,int packedWidth,int padX,int padY)
    {
        constexpr int TW=BX+4,TH=BY+4,NT=BX*BY;
        constexpr float maxCoeff = Feature8 ? 2.0f : 1.0f;
        __shared__ float tile[3][TH][TW];
        __shared__ float4 filters[25*16];
        const int tid=int(threadIdx.y)*BX+int(threadIdx.x);
        const int bx=int(blockIdx.x)*BX,by=int(blockIdx.y)*BY;
        const bool interior=bx>=2 && by>=2 && bx+BX+2<=width && by+BY+2<=height;
        for (int q=tid;q<TW*TH;q+=NT)
        {
            const int ly=q/TW,lx=q-ly*TW;
            int sx=bx+lx-2,sy=by+ly-2;
            if (!interior)
            {
                sx=max(0,min(width-1,sx));
                sy=max(0,min(height-1,sy));
            }
            const ushort4 value=surf2Dread<ushort4>(source,sx*int(sizeof(ushort4)),sy);
            tile[0][ly][lx]=__half2float(__ushort_as_half(value.x));
            tile[1][ly][lx]=__half2float(__ushort_as_half(value.y));
            tile[2][ly][lx]=__half2float(__ushort_as_half(value.z));
        }
        for (int q=tid;q<25*16;q+=NT) filters[q]=basis[q];
        __syncthreads();
        const int x=bx+int(threadIdx.x),y=by+int(threadIdx.y);
        if (x>=width || y>=height) return;
        const int px=x+padX,py=y+padY;
        const int parity=((py&1)<<1)|(px&1);
        const __half* mp=metadata+(size_t(py>>1)*packedWidth+(px>>1))*64+parity*16;
        float coefficients[16];
        #pragma unroll
        for (int k=0;k<16;++k)
            coefficients[k]=fminf(maxCoeff,fmaxf(-maxCoeff,__half2float(mp[k])));
        float4 r=make_float4(0,0,0,0),g=r,b=r;
        #pragma unroll 1
        for (int ry=0;ry<5;++ry)
        {
            const float wy0=ry==1 ? 0.25f : (ry==2 ? 0.75f : 0.0f);
            const float wy1=ry==2 ? 0.75f : (ry==3 ? 0.25f : 0.0f);
            #pragma unroll
            for (int rx=0;rx<5;++rx)
            {
                const float wx0=rx==1 ? 0.25f : (rx==2 ? 0.75f : 0.0f);
                const float wx1=rx==2 ? 0.75f : (rx==3 ? 0.25f : 0.0f);
                // Fold the exact half-pixel bilinear weights into the spatial
                // reconstruction; no separate base texture/pass is required.
                float4 weights=make_float4(wy0*wx0,wy0*wx1,wy1*wx0,wy1*wx1);
                #pragma unroll
                for (int k=0;k<16;++k)
                {
                    const float4 value=filters[(ry*5+rx)*16+k];
                    weights.x=fmaf(coefficients[k],value.x,weights.x);
                    weights.y=fmaf(coefficients[k],value.y,weights.y);
                    weights.z=fmaf(coefficients[k],value.z,weights.z);
                    weights.w=fmaf(coefficients[k],value.w,weights.w);
                }
                const int sy=int(threadIdx.y)+ry,sx=int(threadIdx.x)+rx;
                const float sr=tile[0][sy][sx],sg=tile[1][sy][sx],sb=tile[2][sy][sx];
                r.x=fmaf(weights.x,sr,r.x); r.y=fmaf(weights.y,sr,r.y);
                r.z=fmaf(weights.z,sr,r.z); r.w=fmaf(weights.w,sr,r.w);
                g.x=fmaf(weights.x,sg,g.x); g.y=fmaf(weights.y,sg,g.y);
                g.z=fmaf(weights.z,sg,g.z); g.w=fmaf(weights.w,sg,g.w);
                b.x=fmaf(weights.x,sb,b.x); b.y=fmaf(weights.y,sb,b.y);
                b.z=fmaf(weights.z,sb,b.z); b.w=fmaf(weights.w,sb,b.w);
            }
        }
        const float reds[4]={r.x,r.y,r.z,r.w};
        const float greens[4]={g.x,g.y,g.z,g.w};
        const float blues[4]={b.x,b.y,b.z,b.w};
        #pragma unroll
        for (int phase=0;phase<4;++phase)
        {
            const ushort4 value=make_ushort4(
                __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,reds[phase])))),
                __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,greens[phase])))),
                __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,blues[phase])))),
                __half_as_ushort(__float2half_rn(1.0f)));
            surf2Dwrite(value,output,(x*2+(phase&1))*int(sizeof(ushort4)),y*2+(phase>>1));
        }
    }
    // CUDA 12.1+ large launch parameters supply per-grid constant-cache
    // broadcasts. Each graph owns its 6.4KB bank: no global mutable symbol,
    // basis staging per CTA, slot allocator, or per-frame upload is needed.
    template <int BX=16,int BY=8,bool Feature8=false,bool LearnedBase=false>
    __global__ void ReconstructConstant(cudaSurfaceObject_t source,const __half* __restrict__ metadata,
        const __grid_constant__ std::conditional_t<LearnedBase,LearnedFilterBank,BasisBank> basis,cudaSurfaceObject_t output,
        int width,int height,int packedWidth,int padX,int padY)
    {
        constexpr int TW=BX+4,TH=BY+4,NT=BX*BY;
        __shared__ ushort4 tile[TH][TW];
        const int tid=int(threadIdx.y)*BX+int(threadIdx.x);
        const int bx=int(blockIdx.x)*BX,by=int(blockIdx.y)*BY;
        const bool interior=bx>=2 && by>=2 && bx+BX+2<=width && by+BY+2<=height;
        for (int q=tid;q<TW*TH;q+=NT)
        {
            const int ly=q/TW,lx=q-ly*TW;
            int sx=bx+lx-2,sy=by+ly-2;
            if (!interior)
            {
                sx=max(0,min(width-1,sx));
                sy=max(0,min(height-1,sy));
            }
            tile[ly][lx]=surf2Dread<ushort4>(source,sx*int(sizeof(ushort4)),sy);
        }
        __syncthreads();
        const int x=bx+int(threadIdx.x),y=by+int(threadIdx.y);
        if (x>=width || y>=height) return;
        const int px=x+padX,py=y+padY;
        const int parity=((py&1)<<1)|(px&1);
        const __half* mp=metadata+(size_t(py>>1)*packedWidth+(px>>1))*64+parity*16;
        float coefficients[16];
        #pragma unroll
        for (int group=0;group<2;++group)
        {
            const uint4 packed=reinterpret_cast<const uint4*>(mp)[group];
            #pragma unroll
            for (int j=0;j<4;++j)
            {
                const unsigned bits=j==0 ? packed.x : (j==1 ? packed.y : (j==2 ? packed.z : packed.w));
                coefficients[group*8+j*2]=TransferCoefficient<LearnedBase,Feature8>(
                    __half2float(__ushort_as_half(static_cast<unsigned short>(bits))));
                coefficients[group*8+j*2+1]=TransferCoefficient<LearnedBase,Feature8>(
                    __half2float(__ushort_as_half(static_cast<unsigned short>(bits>>16))));
            }
        }
        float4 r=make_float4(0,0,0,0),g=r,b=r;
        if constexpr (!LearnedBase)
        {
            const int tx=int(threadIdx.x),ty=int(threadIdx.y);
            r=make_float4(BilinearPhase(tile,0,tx,ty,0),BilinearPhase(tile,0,tx,ty,1),
                BilinearPhase(tile,0,tx,ty,2),BilinearPhase(tile,0,tx,ty,3));
            g=make_float4(BilinearPhase(tile,1,tx,ty,0),BilinearPhase(tile,1,tx,ty,1),
                BilinearPhase(tile,1,tx,ty,2),BilinearPhase(tile,1,tx,ty,3));
            b=make_float4(BilinearPhase(tile,2,tx,ty,0),BilinearPhase(tile,2,tx,ty,1),
                BilinearPhase(tile,2,tx,ty,2),BilinearPhase(tile,2,tx,ty,3));
        }
        const int tx=int(threadIdx.x),ty=int(threadIdx.y);
        #pragma unroll 1
        for (int ry=0;ry<5;++ry)
        {
            const int sy=ty+ry;
            const ushort4 p0=tile[sy][tx+0];
            const ushort4 p1=tile[sy][tx+1];
            const ushort4 p2=tile[sy][tx+2];
            const ushort4 p3=tile[sy][tx+3];
            const ushort4 p4=tile[sy][tx+4];

            #pragma unroll
            for (int rx=0;rx<5;++rx)
            {
                float4 weights=make_float4(0,0,0,0);
                if constexpr (LearnedBase) weights=basis.base[ry*5+rx];
                #pragma unroll
                for (int k=0;k<16;++k)
                {
                    const float4 value=basis.values[(ry*5+rx)*16+k];
                    weights.x=fmaf(coefficients[k],value.x,weights.x);
                    weights.y=fmaf(coefficients[k],value.y,weights.y);
                    weights.z=fmaf(coefficients[k],value.z,weights.z);
                    weights.w=fmaf(coefficients[k],value.w,weights.w);
                }
                const ushort4 pix=rx==0 ? p0 : (rx==1 ? p1 : (rx==2 ? p2 : (rx==3 ? p3 : p4)));
                const float sr=__half2float(__ushort_as_half(pix.x));
                const float sg=__half2float(__ushort_as_half(pix.y));
                const float sb=__half2float(__ushort_as_half(pix.z));
                r.x=fmaf(weights.x,sr,r.x); r.y=fmaf(weights.y,sr,r.y);
                r.z=fmaf(weights.z,sr,r.z); r.w=fmaf(weights.w,sr,r.w);
                g.x=fmaf(weights.x,sg,g.x); g.y=fmaf(weights.y,sg,g.y);
                g.z=fmaf(weights.z,sg,g.z); g.w=fmaf(weights.w,sg,g.w);
                b.x=fmaf(weights.x,sb,b.x); b.y=fmaf(weights.y,sb,b.y);
                b.z=fmaf(weights.z,sb,b.z); b.w=fmaf(weights.w,sb,b.w);
            }
        }
        const float reds[4]={r.x,r.y,r.z,r.w};
        const float greens[4]={g.x,g.y,g.z,g.w};
        const float blues[4]={b.x,b.y,b.z,b.w};

        const ushort4 p0 = make_ushort4(
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,reds[0])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,greens[0])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,blues[0])))),
            0x3c00);
        const ushort4 p1 = make_ushort4(
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,reds[1])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,greens[1])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,blues[1])))),
            0x3c00);

        uint4 row0;
        row0.x = (static_cast<uint32_t>(p0.y) << 16) | p0.x;
        row0.y = (static_cast<uint32_t>(p0.w) << 16) | p0.z;
        row0.z = (static_cast<uint32_t>(p1.y) << 16) | p1.x;
        row0.w = (static_cast<uint32_t>(p1.w) << 16) | p1.z;
        surf2Dwrite(row0, output, (x * 2) * int(sizeof(ushort4)), y * 2);

        const ushort4 p2 = make_ushort4(
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,reds[2])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,greens[2])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,blues[2])))),
            0x3c00);
        const ushort4 p3 = make_ushort4(
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,reds[3])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,greens[3])))),
            __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,blues[3])))),
            0x3c00);

        uint4 row1;
        row1.x = (static_cast<uint32_t>(p2.y) << 16) | p2.x;
        row1.y = (static_cast<uint32_t>(p2.w) << 16) | p2.z;
        row1.z = (static_cast<uint32_t>(p3.y) << 16) | p3.x;
        row1.w = (static_cast<uint32_t>(p3.w) << 16) | p3.z;
        surf2Dwrite(row1, output, (x * 2) * int(sizeof(ushort4)), y * 2 + 1);
    }

    template <int BX=16, int BY=8>
    __global__ void ReconstructConstantHalf(
        cudaSurfaceObject_t source,
        const __half* __restrict__ metadata,
        const __grid_constant__ LearnedFilterBankHalf basis,
        cudaSurfaceObject_t output,
        int width, int height, int packedWidth, int padX, int padY)
    {
        constexpr int TW = BX + 4, TH = BY + 4, NT = BX * BY;
        __shared__ ushort4 tile[TH][TW];
        const int tid = int(threadIdx.y) * BX + int(threadIdx.x);
        const int bx = int(blockIdx.x) * BX, by = int(blockIdx.y) * BY;
        const bool interior = bx >= 2 && by >= 2 && bx + BX + 2 <= width && by + BY + 2 <= height;
        
        for (int q = tid; q < TW * TH; q += NT)
        {
            const int ly = q / TW, lx = q - ly * TW;
            int sx = bx + lx - 2, sy = by + ly - 2;
            if (!interior)
            {
                sx = max(0, min(width - 1, sx));
                sy = max(0, min(height - 1, sy));
            }
            tile[ly][lx] = surf2Dread<ushort4>(source, sx * int(sizeof(ushort4)), sy);
        }
        __syncthreads();

        const int x = bx + int(threadIdx.x), y = by + int(threadIdx.y);
        if (x >= width || y >= height) return;
        
        const int px = x + padX, py = y + padY;
        const int parity = ((py & 1) << 1) | (px & 1);
        const __half* mp = metadata + (size_t(py >> 1) * packedWidth + (px >> 1)) * 64 + parity * 16;
        
        __half2 c2[16];
        #pragma unroll
        for (int group = 0; group < 2; ++group)
        {
            const uint4 packed = reinterpret_cast<const uint4*>(mp)[group];
            c2[group * 8 + 0] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.x)));
            c2[group * 8 + 1] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.x >> 16)));
            c2[group * 8 + 2] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.y)));
            c2[group * 8 + 3] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.y >> 16)));
            c2[group * 8 + 4] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.z)));
            c2[group * 8 + 5] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.z >> 16)));
            c2[group * 8 + 6] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.w)));
            c2[group * 8 + 7] = MakeCoeff2(__ushort_as_half(static_cast<unsigned short>(packed.w >> 16)));
        }

        const int tx = int(threadIdx.x), ty = int(threadIdx.y);
        
        __half2 r_xy = make_half2(0.0f, 0.0f);
        __half2 r_zw = make_half2(0.0f, 0.0f);
        __half2 g_xy = make_half2(0.0f, 0.0f);
        __half2 g_zw = make_half2(0.0f, 0.0f);
        __half2 b_xy = make_half2(0.0f, 0.0f);
        __half2 b_zw = make_half2(0.0f, 0.0f);

        #pragma unroll 1
        for (int ry = 0; ry < 5; ++ry)
        {
            const int sy = ty + ry;
            const ushort4 p0 = tile[sy][tx + 0];
            const ushort4 p1 = tile[sy][tx + 1];
            const ushort4 p2 = tile[sy][tx + 2];
            const ushort4 p3 = tile[sy][tx + 3];
            const ushort4 p4 = tile[sy][tx + 4];

            #pragma unroll
            for (int rx = 0; rx < 5; ++rx)
            {
                Half4 weights = basis.base[ry * 5 + rx];
                
                #pragma unroll
                for (int k = 0; k < 16; ++k)
                {
                    Half4 val = basis.values[(ry * 5 + rx) * 16 + k];
                    weights.xy = __hfma2(c2[k], val.xy, weights.xy);
                    weights.zw = __hfma2(c2[k], val.zw, weights.zw);
                }

                const ushort4 pix = (rx == 0) ? p0 : ((rx == 1) ? p1 : ((rx == 2) ? p2 : ((rx == 3) ? p3 : p4)));
                const __half2 sr2 = __half2half2(__ushort_as_half(pix.x));
                const __half2 sg2 = __half2half2(__ushort_as_half(pix.y));
                const __half2 sb2 = __half2half2(__ushort_as_half(pix.z));

                r_xy = __hfma2(weights.xy, sr2, r_xy);
                r_zw = __hfma2(weights.zw, sr2, r_zw);
                g_xy = __hfma2(weights.xy, sg2, g_xy);
                g_zw = __hfma2(weights.zw, sg2, g_zw);
                b_xy = __hfma2(weights.xy, sb2, b_xy);
                b_zw = __hfma2(weights.zw, sb2, b_zw);
            }
        }

        const __half2 zero2 = make_half2(0.0f, 0.0f);
        const __half2 one2  = make_half2(1.0f, 1.0f);

        r_xy = __hmin2(one2, __hmax2(zero2, r_xy));
        r_zw = __hmin2(one2, __hmax2(zero2, r_zw));
        g_xy = __hmin2(one2, __hmax2(zero2, g_xy));
        g_zw = __hmin2(one2, __hmax2(zero2, g_zw));
        b_xy = __hmin2(one2, __hmax2(zero2, b_xy));
        b_zw = __hmin2(one2, __hmax2(zero2, b_zw));

        const ushort4 p0 = make_ushort4(__half_as_ushort(r_xy.x), __half_as_ushort(g_xy.x), __half_as_ushort(b_xy.x), 0x3c00);
        const ushort4 p1 = make_ushort4(__half_as_ushort(r_xy.y), __half_as_ushort(g_xy.y), __half_as_ushort(b_xy.y), 0x3c00);
        const ushort4 p2 = make_ushort4(__half_as_ushort(r_zw.x), __half_as_ushort(g_zw.x), __half_as_ushort(b_zw.x), 0x3c00);
        const ushort4 p3 = make_ushort4(__half_as_ushort(r_zw.y), __half_as_ushort(g_zw.y), __half_as_ushort(b_zw.y), 0x3c00);

        uint4 row0;
        row0.x = (static_cast<uint32_t>(p0.y) << 16) | p0.x;
        row0.y = (static_cast<uint32_t>(p0.w) << 16) | p0.z;
        row0.z = (static_cast<uint32_t>(p1.y) << 16) | p1.x;
        row0.w = (static_cast<uint32_t>(p1.w) << 16) | p1.z;
        surf2Dwrite(row0, output, (x * 2) * int(sizeof(ushort4)), y * 2);

        uint4 row1;
        row1.x = (static_cast<uint32_t>(p2.y) << 16) | p2.x;
        row1.y = (static_cast<uint32_t>(p2.w) << 16) | p2.z;
        row1.z = (static_cast<uint32_t>(p3.y) << 16) | p3.x;
        row1.w = (static_cast<uint32_t>(p3.w) << 16) | p3.z;
        surf2Dwrite(row1, output, (x * 2) * int(sizeof(ushort4)), y * 2 + 1);
    }

    template <int BX=8,int BY=32>
    __global__ void ReconstructPairMoEConstant(
        cudaSurfaceObject_t source,
        const __half* __restrict__ coeffMeta,
        const __half* __restrict__ routeMeta,
        const __grid_constant__ LearnedPairMoEFilterBank basis,
        cudaSurfaceObject_t output,
        int width,int height,int packedWidth,int padX,int padY)
    {
        constexpr int TW=BX+4,TH=BY+4,NT=BX*BY;
        __shared__ float tile[3][TH][TW];
        __shared__ float4 s_basis[25*32];
        const int tid=int(threadIdx.y)*BX+int(threadIdx.x);
        const int bx=int(blockIdx.x)*BX,by=int(blockIdx.y)*BY;
        const bool interior=bx>=2 && by>=2 && bx+BX+2<=width && by+BY+2<=height;
        for (int q=tid;q<TW*TH;q+=NT)
        {
            const int ly=q/TW,lx=q-ly*TW;
            int sx=bx+lx-2,sy=by+ly-2;
            if (!interior)
            {
                sx=max(0,min(width-1,sx));
                sy=max(0,min(height-1,sy));
            }
            const ushort4 value=surf2Dread<ushort4>(source,sx*int(sizeof(ushort4)),sy);
            tile[0][ly][lx]=__half2float(__ushort_as_half(value.x));
            tile[1][ly][lx]=__half2float(__ushort_as_half(value.y));
            tile[2][ly][lx]=__half2float(__ushort_as_half(value.z));
        }
        for (int q=tid;q<25*32;q+=NT)
            s_basis[q]=basis.values[q];
        __syncthreads();
        const int x=bx+int(threadIdx.x),y=by+int(threadIdx.y);
        if (x>=width || y>=height) return;
        const int px=x+padX,py=y+padY;
        const int parity=((py&1)<<1)|(px&1);
        const size_t offset=(size_t(py>>1)*packedWidth+(px>>1))*64+parity*16;
        const __half* mp_a=coeffMeta+offset;
        const __half* mp_r=routeMeta+offset;
        float a[16];
        int route[16];
        #pragma unroll
        for (int group=0;group<2;++group)
        {
            const uint4 packed_a=reinterpret_cast<const uint4*>(mp_a)[group];
            const uint4 packed_r=reinterpret_cast<const uint4*>(mp_r)[group];
            #pragma unroll
            for (int j=0;j<4;++j)
            {
                const unsigned bits_a=j==0 ? packed_a.x : (j==1 ? packed_a.y : (j==2 ? packed_a.z : packed_a.w));
                const unsigned bits_r=j==0 ? packed_r.x : (j==1 ? packed_r.y : (j==2 ? packed_r.z : packed_r.w));
                const float a0=TransferCoefficient<true,true>(__half2float(__ushort_as_half(static_cast<unsigned short>(bits_a))));
                const float a1=TransferCoefficient<true,true>(__half2float(__ushort_as_half(static_cast<unsigned short>(bits_a>>16))));
                const float r0=__half2float(__ushort_as_half(static_cast<unsigned short>(bits_r)));
                const float r1=__half2float(__ushort_as_half(static_cast<unsigned short>(bits_r>>16)));
                a[group*8+j*2]=a0;
                a[group*8+j*2+1]=a1;
                route[group*8+j*2]=r0>=0.0f ? 1 : 0;
                route[group*8+j*2+1]=r1>=0.0f ? 1 : 0;
            }
        }
        float4 r=make_float4(0,0,0,0),g=r,b=r;
        const int tx=int(threadIdx.x),ty=int(threadIdx.y);
        #pragma unroll 1
        for (int ry=0;ry<5;++ry)
        {
            const int sy=ty+ry;
            const float p0_r=tile[0][sy][tx+0],p0_g=tile[1][sy][tx+0],p0_b=tile[2][sy][tx+0];
            const float p1_r=tile[0][sy][tx+1],p1_g=tile[1][sy][tx+1],p1_b=tile[2][sy][tx+1];
            const float p2_r=tile[0][sy][tx+2],p2_g=tile[1][sy][tx+2],p2_b=tile[2][sy][tx+2];
            const float p3_r=tile[0][sy][tx+3],p3_g=tile[1][sy][tx+3],p3_b=tile[2][sy][tx+3];
            const float p4_r=tile[0][sy][tx+4],p4_g=tile[1][sy][tx+4],p4_b=tile[2][sy][tx+4];

            #pragma unroll
            for (int rx=0;rx<5;++rx)
            {
                const int t=ry*5+rx;
                float4 weights=basis.base[t];
                #pragma unroll
                for (int k=0;k<16;++k)
                {
                    const int expert=(k<<1)+route[k];
                    const float4 value=s_basis[t*32+expert];
                    weights.x=fmaf(a[k],value.x,weights.x);
                    weights.y=fmaf(a[k],value.y,weights.y);
                    weights.z=fmaf(a[k],value.z,weights.z);
                    weights.w=fmaf(a[k],value.w,weights.w);
                }
                const float sr=rx==0 ? p0_r : (rx==1 ? p1_r : (rx==2 ? p2_r : (rx==3 ? p3_r : p4_r)));
                const float sg=rx==0 ? p0_g : (rx==1 ? p1_g : (rx==2 ? p2_g : (rx==3 ? p3_g : p4_g)));
                const float sb=rx==0 ? p0_b : (rx==1 ? p1_b : (rx==2 ? p2_b : (rx==3 ? p3_b : p4_b)));
                r.x=fmaf(weights.x,sr,r.x); r.y=fmaf(weights.y,sr,r.y);
                r.z=fmaf(weights.z,sr,r.z); r.w=fmaf(weights.w,sr,r.w);
                g.x=fmaf(weights.x,sg,g.x); g.y=fmaf(weights.y,sg,g.y);
                g.z=fmaf(weights.z,sg,g.z); g.w=fmaf(weights.w,sg,g.w);
                b.x=fmaf(weights.x,sb,b.x); b.y=fmaf(weights.y,sb,b.y);
                b.z=fmaf(weights.z,sb,b.z); b.w=fmaf(weights.w,sb,b.w);
            }
        }
        const float reds[4]={r.x,r.y,r.z,r.w};
        const float greens[4]={g.x,g.y,g.z,g.w};
        const float blues[4]={b.x,b.y,b.z,b.w};
        #pragma unroll
        for (int phase=0;phase<4;++phase)
        {
            const ushort4 value=make_ushort4(
                __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,reds[phase])))),
                __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,greens[phase])))),
                __half_as_ushort(__float2half_rn(fminf(1.0f,fmaxf(0.0f,blues[phase])))),
                0x3c00);
            surf2Dwrite(value,output,(x*2+(phase&1))*int(sizeof(ushort4)),y*2+(phase>>1));
        }
    }
}
