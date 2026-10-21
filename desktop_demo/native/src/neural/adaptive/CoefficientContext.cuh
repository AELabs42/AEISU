#pragma once

// Experimental 64-token positive linear attention. No dependency on reconstruction.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace aess::adaptive::context
{
    struct Parameters
    {
        __half matrix[256]{}; // Row-major symmetric square root of trace-normalized Gram.
        float gamma = 0.0f;
        float beta = 1.0f;
        float epsilon = 1.0e-6f;
    };

    __host__ __device__ inline float Transfer(float a)
    {
        const float magnitude = fabsf(a);
        return copysignf(magnitude <= 2.0f ? magnitude :
            fminf(2.5f, fmaf(0.05f, magnitude - 2.0f, 2.0f)), a);
    }

    __device__ inline float Inverse(float u)
    {
        const float magnitude = fabsf(u);
        return copysignf(magnitude <= 2.0f ? magnitude :
            fminf(12.0f, fmaf(magnitude - 2.0f, 20.0f, 2.0f)), u);
    }

    __device__ inline size_t Offset(int x, int y, int packedWidth, int padX, int padY)
    {
        const int px = x + padX, py = y + padY;
        return (size_t(py >> 1) * packedWidth + (px >> 1)) * 64 +
            (((py & 1) << 1) | (px & 1)) * 16;
    }

    // Four warps per CTA. Stage 1: each warp transforms 16 tokens.
    // Stage 2: each warp forms a 16-token partial P^T U, summed in FP32.
    // Stage 3: each warp forms 16 output tokens. All three products use WMMA.
    __global__ void CoefficientContextKernel(const __half* __restrict__ input,
        __half* __restrict__ output, int width, int height, int packedWidth,
        int padX, int padY, const __grid_constant__ Parameters parameters)
    {
        __shared__ __align__(32) __half raw[1024], values[1024], positive[1024];
        __shared__ __align__(32) __half matrix[256], statistic[256];
        __shared__ __align__(32) float work[4][256];
        __shared__ float sum[16], minimum[16], maximum[16], denominator[64];
        const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
        const int bx = int(blockIdx.x) * 8, by = int(blockIdx.y) * 8;
        const int validWidth = min(8, width - bx), validHeight = min(8, height - by);
        const int count = validWidth * validHeight;
        for (int i = tid; i < 1024; i += 128)
        {
            const int token = i / 16, x = token % 8, y = token / 8;
            const bool valid = x < validWidth && y < validHeight;
            const __half a = valid ? input[Offset(bx+x, by+y, packedWidth, padX, padY)+i%16] : __float2half(0.0f);
            raw[i] = a;
            values[i] = __float2half_rn(Transfer(__half2float(a)));
        }
        for (int i = tid; i < 256; i += 128) matrix[i] = parameters.matrix[i];
        __syncthreads();
        // Reduce eight-token partials across the whole CTA instead of leaving
        // 112 threads idle while 16 threads serially scan all 64 tokens.
        bool changed = false;
        float lo = 2.5f, hi = -2.5f;
        const unsigned short first = __half_as_ushort(raw[tid%16]);
        for (int i = tid; i < 1024; i += 128)
        {
            const int token = i / 16;
            if (token%8 >= validWidth || token/8 >= validHeight) continue;
            changed |= __half_as_ushort(raw[i]) != first;
            const float u = Transfer(__half2float(raw[i]));
            lo = fminf(lo, u); hi = fmaxf(hi, u);
        }
        work[0][tid] = lo;
        work[0][128+tid] = hi;
        __syncthreads();
        if (tid < 16)
        {
            lo = 2.5f; hi = -2.5f;
            #pragma unroll
            for (int group = 0; group < 8; ++group)
            {
                lo = fminf(lo, work[0][group*16+tid]);
                hi = fmaxf(hi, work[0][128+group*16+tid]);
            }
            minimum[tid] = lo; maximum[tid] = hi;
        }
        const bool constant = __syncthreads_or(changed) == 0;
        if (constant)
        {
            // Exact constant-field invariant, including saturated/raw signed-zero values.
            for (int i = tid; i < 1024; i += 128)
            {
                const int token = i/16, x = token%8, y = token/8;
                if (x < validWidth && y < validHeight)
                    output[Offset(bx+x, by+y, packedWidth, padX, padY)+i%16] = raw[i];
            }
            return;
        }
        using namespace nvcuda;
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::load_matrix_sync(a, values+warp*256, 16);
        wmma::load_matrix_sync(b, matrix, 16);
        wmma::fill_fragment(c, 0.0f);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(work[warp], c, 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32)
        {
            const int token = warp*16+i/16;
            const bool valid = token%8 < validWidth && token/8 < validHeight;
            positive[warp*256+i] = __float2half_rn(valid ?
                1.0f+fmaxf(0.0f, parameters.beta*work[warp][i]) : 0.0f);
        }
        __syncthreads();
        float r = 0.0f;
        for (int i = tid; i < 1024; i += 128) r += __half2float(positive[i]);
        work[0][tid] = r;
        __syncthreads();
        if (tid < 16)
        {
            r = 0.0f;
            #pragma unroll
            for (int group = 0; group < 8; ++group) r += work[0][group*16+tid];
            sum[tid] = r / float(count);
        }
        // Warp zero must finish reading its shared partial sums before any
        // lane reuses that storage for the next WMMA accumulator.
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> pt;
        wmma::load_matrix_sync(pt, positive+warp*256, 16);
        wmma::load_matrix_sync(b, values+warp*256, 16);
        wmma::fill_fragment(c, 0.0f);
        wmma::mma_sync(c, pt, b, c);
        wmma::store_matrix_sync(work[warp], c, 16, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < 256; i += 128)
        {
            float s = 0.0f;
            for (int j = 0; j < 4; ++j) s += work[j][i];
            // Store S/count in FP16 to avoid overflowing the intermediate.
            // r and epsilon use the same scaling, preserving the real-valued formula.
            statistic[i] = __float2half_rn(s / float(count));
        }
        if (tid < 64)
        {
            float d = parameters.epsilon / float(count);
            for (int k = 0; k < 16; ++k) d = fmaf(__half2float(positive[tid*16+k]), sum[k], d);
            denominator[tid] = d;
        }
        __syncthreads();
        wmma::load_matrix_sync(a, positive+warp*256, 16);
        wmma::load_matrix_sync(b, statistic, 16);
        wmma::fill_fragment(c, 0.0f);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(work[warp], c, 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32)
        {
            const int token = warp*16+i/16, k = i%16, x = token%8, y = token/8;
            if (x >= validWidth || y >= validHeight) continue;
            const float u = Transfer(__half2float(raw[token*16+k]));
            const float yValue = fminf(maximum[k], fmaxf(minimum[k], work[warp][i]/denominator[token]));
            const float mixed = fmaf(parameters.gamma, yValue-u, u);
            output[Offset(bx+x, by+y, packedWidth, padX, padY)+k] = __float2half_rn(Inverse(mixed));
        }
    }

    inline void Validate(const Parameters& p)
    {
        if (!std::isfinite(p.gamma) || p.gamma < 0 || p.gamma > 1 ||
            !std::isfinite(p.beta) || p.beta < 0 || p.beta > 16 ||
            !std::isfinite(p.epsilon) || p.epsilon <= 0)
            throw std::invalid_argument("Context requires gamma in [0,1], beta in [0,16], epsilon > 0");
        float norm = 0;
        for (const auto value : p.matrix)
        {
            const float f = __half2float(value);
            if (!std::isfinite(f)) throw std::invalid_argument("Nonfinite context matrix");
            norm += f*f;
        }
        if (norm < 15.8f || norm > 16.2f)
            throw std::invalid_argument("Context matrix must be the trace-16 normalized Gram square root");
    }

    inline cudaError_t Launch(const __half* source, __half* destination, int width, int height,
        int packedWidth, int packedHeight, int padX, int padY, const Parameters& p, cudaStream_t stream,
        bool enable = true)
    {
        if (!source || !destination || width <= 0 || height <= 0 || packedWidth <= 0 || packedHeight <= 0 ||
            padX < 0 || padY < 0 || int64_t(width)+padX > int64_t(packedWidth)*2 ||
            int64_t(height)+padY > int64_t(packedHeight)*2) return cudaErrorInvalidValue;
        if (!enable || p.gamma == 0)
            return source == destination ? cudaSuccess : cudaMemcpyAsync(destination, source,
                size_t(packedWidth)*packedHeight*64*sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        if (source == destination) return cudaErrorInvalidValue;
        CoefficientContextKernel<<<dim3((width+7)/8,(height+7)/8),128,0,stream>>>(
            source,destination,width,height,packedWidth,padX,padY,p);
        return cudaGetLastError();
    }

    struct Settings
    {
        Parameters parameters{};
        bool enabled = false;
        bool Active() const { return enabled && parameters.gamma != 0; }
    };

    inline Settings ReadEnvironment()
    {
        Settings settings;
        const char* enabled = std::getenv("AESS_CONTEXT_ENABLE");
        settings.enabled = enabled && std::string(enabled) == "1";
        if (!settings.enabled) return settings;
        const char* gamma = std::getenv("AESS_CONTEXT_GAMMA");
        const char* beta = std::getenv("AESS_CONTEXT_BETA");
        settings.parameters.gamma = gamma ? std::stof(gamma) : 0.125f;
        settings.parameters.beta = beta ? std::stof(beta) : 1.0f;
        const char* path = std::getenv("AESS_CONTEXT_MATRIX");
        if (!path) throw std::invalid_argument("AESS_CONTEXT_MATRIX must identify a 512-byte FP16 L matrix");
        std::ifstream input(path, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(settings.parameters.matrix),512) || input.peek() != EOF)
            throw std::runtime_error("Context matrix must contain exactly 256 FP16 entries");
        Validate(settings.parameters);
        return settings;
    }
}
