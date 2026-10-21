#include "SrcPyrCudaRuntime.h"
#include "SrcPyrCudnnConv.h"
#include "SrcPyrModel.h"
#include "SrcPyrBlackwellFp4Conv.cuh"
#include "SrcPyrNativeFp8Conv.cuh"
#include "SrcPyrNativeFp16Refine.cuh"
#include "SrcPyrNativeFp8Expand.cuh"
#include "SrcPyrNativeFp8Compress.cuh"
#include "../NeuralInterop.h"
#include "../adaptive/AdaptiveReconstruction.cuh"
#include "../adaptive/DirectRgbReconstruction.cuh"
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
#include "../adaptive/CoefficientContext.cuh"
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include <cuda_d3d11_interop.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublasLt.h>

#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace aess::srcpyr
{
    namespace
    {
        using Fp8 = cutlass::float_e4m3_t;
        using Half = cutlass::half_t;
        using Nhwc = cutlass::layout::TensorNHWC;
        using ConvEpilogue = cutlass::epilogue::thread::LinearCombination<
            float, 4, float, float>;

        // SM89 is CUTLASS 2.x's native E4M3 Tensor Core convolution contract.
        // nvcc emits this kernel into the application's SM120 image, so there
        // is no per-launch conversion or framework dispatch in the hot path.
        using Fp8ConvKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
            Fp8, Nhwc, Fp8, Nhwc, float, Nhwc, float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm89,
            cutlass::gemm::GemmShape<128, 64, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>,
            cutlass::gemm::GemmShape<16, 8, 32>,
            ConvEpilogue,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            3,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided>::Kernel;
        using Fp8ConvOperation = cutlass::conv::device::ImplicitGemmConvolution<Fp8ConvKernel>;

        using HalfConvKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
            Half, Nhwc, Half, Nhwc, float, Nhwc, float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 64, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            ConvEpilogue,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            3,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided>::Kernel;
        using HalfConvOperation = cutlass::conv::device::ImplicitGemmConvolution<HalfConvKernel>;

        void CheckCuda(cudaError_t status, const char* operation)
        {
            if (status != cudaSuccess)
                throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
        }

        void CheckCutlass(cutlass::Status status, const char* operation)
        {
            if (status != cutlass::Status::kSuccess)
                throw std::runtime_error(std::string(operation) + " failed with CUTLASS status " +
                    std::to_string(static_cast<int>(status)));
        }

        void CheckCublas(cublasStatus_t status, const char* operation)
        {
            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                    std::to_string(static_cast<int>(status)));
        }

        inline void CheckLaunch(const char* operation)
        {
#if !defined(NDEBUG) || defined(AESS_SRCPYR_VALIDATE_LAUNCHES)
            CheckCuda(cudaGetLastError(), operation);
#else
            (void)operation;
#endif
        }

        float EventElapsed(cudaEvent_t begin, cudaEvent_t end)
        {
            float value = 0.0f;
            cudaError_t status = cudaEventElapsedTime(&value, begin, end);
            if (status == cudaErrorNotReady)
            {
                CheckCuda(cudaEventSynchronize(end), "cudaEventSynchronize(srcpyr timing)");
                status = cudaEventElapsedTime(&value, begin, end);
            }
            CheckCuda(status, "cudaEventElapsedTime(srcpyr)");
            return value;
        }

        enum class EasuKernelMode
        {
            TiledFp16,
            ReferenceFp32,
        };

        EasuKernelMode ReadEasuKernelMode()
        {
            std::array<char,64> value{};
            const DWORD length = GetEnvironmentVariableA(
                "AESS_SRCPYR_EASU_KERNEL", value.data(),
                static_cast<DWORD>(value.size()));
            if (length == 0) return EasuKernelMode::TiledFp16;
            if (length >= value.size())
                throw std::runtime_error(
                    "AESS_SRCPYR_EASU_KERNEL value is too long");
            std::string normalized(value.data(), length);
            std::transform(
                normalized.begin(), normalized.end(), normalized.begin(),
                [](unsigned char character)
                {
                    return static_cast<char>(std::tolower(character));
                });
            if (normalized == "tiled-fp16" || normalized == "fp16" ||
                normalized == "optimized")
                return EasuKernelMode::TiledFp16;
            if (normalized == "reference-fp32" || normalized == "fp32" ||
                normalized == "reference")
                return EasuKernelMode::ReferenceFp32;
            throw std::runtime_error(
                "AESS_SRCPYR_EASU_KERNEL must be tiled-fp16 or reference-fp32");
        }

        bool UseNativeFp16Refine()
        {
            std::array<char, 16> value{};
            const DWORD length = GetEnvironmentVariableA(
                "AESS_SRCPYR_REFINE_BACKEND", value.data(),
                static_cast<DWORD>(value.size()));
            if (length == 0) return true;
            if (length >= value.size())
                throw std::runtime_error(
                    "AESS_SRCPYR_REFINE_BACKEND value is too long");
            std::string normalized(value.data(), length);
            std::transform(
                normalized.begin(), normalized.end(), normalized.begin(),
                [](unsigned char character)
                {
                    return static_cast<char>(std::tolower(character));
                });
            if (normalized == "native") return true;
            if (normalized == "cudnn") return false;
            throw std::runtime_error(
                "AESS_SRCPYR_REFINE_BACKEND must be native or cudnn");
        }

        bool UseFusedTail()
        {
            std::array<char, 16> value{};
            const DWORD length = GetEnvironmentVariableA(
                "AESS_SRCPYR_FUSED_TAIL", value.data(),
                static_cast<DWORD>(value.size()));
            if (length == 0) return true;
            if (length >= value.size())
                throw std::runtime_error(
                    "AESS_SRCPYR_FUSED_TAIL value is too long");
            std::string normalized(value.data(), length);
            std::transform(
                normalized.begin(), normalized.end(), normalized.begin(),
                [](unsigned char character)
                {
                    return static_cast<char>(std::tolower(character));
                });
            if (normalized == "true" || normalized == "1" || normalized == "fused") return true;
            if (normalized == "false" || normalized == "0" || normalized == "separate") return false;
            throw std::runtime_error(
                "AESS_SRCPYR_FUSED_TAIL must be true, false, 1, 0, fused, or separate");
        }

        bool UseFusedProject()
        {
            std::array<char, 16> value{};
            const DWORD length = GetEnvironmentVariableA(
                "AESS_SRCPYR_FUSED_PROJECT", value.data(),
                static_cast<DWORD>(value.size()));
            if (length == 0) return true;
            if (length >= value.size())
                throw std::runtime_error(
                    "AESS_SRCPYR_FUSED_PROJECT value is too long");
            std::string normalized(value.data(), length);
            std::transform(
                normalized.begin(), normalized.end(), normalized.begin(),
                [](unsigned char character)
                {
                    return static_cast<char>(std::tolower(character));
                });
            if (normalized == "true" || normalized == "1" || normalized == "fused") return true;
            if (normalized == "false" || normalized == "0" || normalized == "separate") return false;
            throw std::runtime_error(
                "AESS_SRCPYR_FUSED_PROJECT must be true, false, 1, 0, fused, or separate");
        }

        const char* EasuKernelModeName(EasuKernelMode mode) noexcept
        {
            if (mode == EasuKernelMode::TiledFp16)
                return "tiled-exact2x-fp16x2";
            return "reference-fp32";
        }

        class DeviceBuffer
        {
        public:
            DeviceBuffer() = default;
            ~DeviceBuffer() { Reset(); }
            DeviceBuffer(const DeviceBuffer&) = delete;
            DeviceBuffer& operator=(const DeviceBuffer&) = delete;
            void Allocate(size_t bytes)
            {
                Reset();
                if (bytes != 0)
                {
                    CheckCuda(cudaMalloc(&m_data, bytes), "cudaMalloc(srcpyr)");
                    m_bytes = bytes;
                }
            }
            void Reset() noexcept
            {
                if (m_data != nullptr) cudaFree(m_data);
                m_data = nullptr;
                m_bytes = 0;
            }
            template <typename T> T* As() const noexcept { return static_cast<T*>(m_data); }
            void* Get() const noexcept { return m_data; }
            size_t Bytes() const noexcept { return m_bytes; }
        private:
            void* m_data = nullptr;
            size_t m_bytes = 0;
        };

        class ExternalD3D11Resource
        {
        public:
            ExternalD3D11Resource() = default;
            ~ExternalD3D11Resource() { Reset(); }
            ExternalD3D11Resource(const ExternalD3D11Resource&) = delete;
            ExternalD3D11Resource& operator=(const ExternalD3D11Resource&) = delete;
            void Import(ID3D11Texture2D* texture, uint32_t width, uint32_t height)
            {
                Reset();
                D3D11_TEXTURE2D_DESC textureDesc{};
                texture->GetDesc(&textureDesc);
                ID3D12Resource* sharedAllocation=nullptr;
                UINT privateSize=sizeof(sharedAllocation);
                if (SUCCEEDED(texture->GetPrivateData(aess::neural::kCudaSharedResource,&privateSize,&sharedAllocation)))
                {
                    Microsoft::WRL::ComPtr<ID3D12Resource> owner;
                    owner.Attach(sharedAllocation);
                    Microsoft::WRL::ComPtr<ID3D12Device> device12;
                    HRESULT result=sharedAllocation->GetDevice(IID_PPV_ARGS(&device12));
                    if (FAILED(result)) throw std::runtime_error("Get shared texture allocation device failed");
                    const auto resource=sharedAllocation->GetDesc();
                    const auto footprint=device12->GetResourceAllocationInfo(0,1,&resource);
                    HANDLE shared=nullptr;
                    result=device12->CreateSharedHandle(sharedAllocation,nullptr,GENERIC_ALL,nullptr,&shared);
                    if (FAILED(result)) throw std::runtime_error("Create CUDA shared allocation handle failed");
                    cudaExternalMemoryHandleDesc exact{};
                    exact.type=cudaExternalMemoryHandleTypeD3D12Resource;
                    exact.handle.win32.handle=shared;
                    exact.size=footprint.SizeInBytes;
                    exact.flags=cudaExternalMemoryDedicated;
                    const auto imported=cudaImportExternalMemory(&m_memory,&exact);
                    CloseHandle(shared);
                    CheckCuda(imported,"cudaImportExternalMemory(exact shared allocation)");
                    MapArray(width,height);
                    return;
                }
                HANDLE handle = nullptr;
                bool closeHandle = false;
                cudaExternalMemoryHandleDesc desc{};
                HRESULT hr = E_FAIL;
                if ((textureDesc.MiscFlags&D3D11_RESOURCE_MISC_SHARED_NTHANDLE)!=0)
                {
                    IDXGIResource1* resource = nullptr;
                    hr = texture->QueryInterface(
                        __uuidof(IDXGIResource1),reinterpret_cast<void**>(&resource));
                    if (SUCCEEDED(hr) && resource != nullptr)
                    {
                        hr = resource->CreateSharedHandle(
                            nullptr,DXGI_SHARED_RESOURCE_READ|DXGI_SHARED_RESOURCE_WRITE,
                            nullptr,&handle);
                        resource->Release();
                    }
                    desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
                    closeHandle = true;
                }
                else
                {
                    IDXGIResource* resource = nullptr;
                    hr = texture->QueryInterface(
                        __uuidof(IDXGIResource),reinterpret_cast<void**>(&resource));
                    if (SUCCEEDED(hr) && resource != nullptr)
                    {
                        hr = resource->GetSharedHandle(&handle);
                        resource->Release();
                    }
                    desc.type = cudaExternalMemoryHandleTypeD3D11ResourceKmt;
                }
                if (FAILED(hr) || handle == nullptr)
                    throw std::runtime_error("srcpyr could not obtain a D3D11 shared handle");
                desc.handle.win32.handle = handle;
                // A D3D11 shared texture is tiled and its allocation can be
                // larger than aligned width*height*bytes-per-pixel. Passing
                // the logical byte count works for many common resolutions
                // but makes some valid sizes fault in later tiles. Ask D3D12
                // for the adapter's authoritative shared-resource allocation
                // footprint instead; the standard shared layout is the same
                // cross-API contract used by CUDA's D3D11 KMT import.
                ID3D11Device* d3d11Device = nullptr;
                texture->GetDevice(&d3d11Device);
                IDXGIDevice* dxgiDevice = nullptr;
                HRESULT sizeHr = d3d11Device->QueryInterface(
                    __uuidof(IDXGIDevice),reinterpret_cast<void**>(&dxgiDevice));
                d3d11Device->Release();
                IDXGIAdapter* adapter = nullptr;
                if (SUCCEEDED(sizeHr) && dxgiDevice != nullptr)
                {
                    sizeHr = dxgiDevice->GetAdapter(&adapter);
                    dxgiDevice->Release();
                }
                ID3D12Device* d3d12Device = nullptr;
                if (SUCCEEDED(sizeHr) && adapter != nullptr)
                    sizeHr = D3D12CreateDevice(
                        adapter,D3D_FEATURE_LEVEL_11_0,
                        __uuidof(ID3D12Device),
                        reinterpret_cast<void**>(&d3d12Device));
                if (adapter != nullptr) adapter->Release();
                if (FAILED(sizeHr) || d3d12Device == nullptr)
                    throw std::runtime_error(
                        "SourcePyramid could not query the shared texture allocation size");
                D3D12_RESOURCE_DESC resourceDesc{};
                resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                resourceDesc.Width = width;
                resourceDesc.Height = height;
                resourceDesc.DepthOrArraySize = 1;
                resourceDesc.MipLevels = 1;
                resourceDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                resourceDesc.SampleDesc.Count = 1;
                resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
                resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
                const D3D12_RESOURCE_ALLOCATION_INFO allocation =
                    d3d12Device->GetResourceAllocationInfo(0,1,&resourceDesc);
                d3d12Device->Release();
                if (allocation.SizeInBytes == 0 ||
                    allocation.SizeInBytes == std::numeric_limits<UINT64>::max())
                    throw std::runtime_error(
                        "SourcePyramid received an invalid shared texture allocation size");
                desc.size = allocation.SizeInBytes;
                desc.flags = cudaExternalMemoryDedicated;
                const cudaError_t importStatus = cudaImportExternalMemory(&m_memory,&desc);
                if (closeHandle) CloseHandle(handle);
                CheckCuda(importStatus,"cudaImportExternalMemory(srcpyr)");
                MapArray(width,height);
            }
            void MapArray(uint32_t width,uint32_t height)
            {
                cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
                arrayDesc.formatDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
                arrayDesc.extent = make_cudaExtent(width, height, 0);
                arrayDesc.flags = cudaArraySurfaceLoadStore | cudaArrayColorAttachment;
                arrayDesc.numLevels = 1;
                CheckCuda(cudaExternalMemoryGetMappedMipmappedArray(&m_mipmap, m_memory, &arrayDesc),
                    "cudaExternalMemoryGetMappedMipmappedArray(srcpyr)");
                CheckCuda(cudaGetMipmappedArrayLevel(&m_array, m_mipmap, 0), "cudaGetMipmappedArrayLevel(srcpyr)");
                cudaResourceDesc surfaceDesc{};
                surfaceDesc.resType = cudaResourceTypeArray;
                surfaceDesc.res.array.array = m_array;
                CheckCuda(cudaCreateSurfaceObject(&m_surface, &surfaceDesc), "cudaCreateSurfaceObject(srcpyr)");
            }
            void Reset() noexcept
            {
                if (m_surface != 0) cudaDestroySurfaceObject(m_surface);
                if (m_mipmap != nullptr) cudaFreeMipmappedArray(m_mipmap);
                if (m_memory != nullptr) cudaDestroyExternalMemory(m_memory);
                m_surface = 0; m_array = nullptr; m_mipmap = nullptr; m_memory = nullptr;
            }
            cudaSurfaceObject_t Surface() const noexcept { return m_surface; }
        private:
            cudaExternalMemory_t m_memory = nullptr;
            cudaMipmappedArray_t m_mipmap = nullptr;
            cudaArray_t m_array = nullptr;
            cudaSurfaceObject_t m_surface = 0;
        };

        __device__ __forceinline__ int ClampInt(int value, int low, int high)
        {
            return value < low ? low : (value > high ? high : value);
        }

        __device__ __forceinline__ float3 ReadSource(
            cudaSurfaceObject_t source, int width, int height, int y, int x)
        {
            x = ClampInt(x, 0, width - 1); y = ClampInt(y, 0, height - 1);
            const ushort4 packed = surf2Dread<ushort4>(source, x * int(sizeof(ushort4)), y);
            return make_float3(__half2float(__ushort_as_half(packed.x)),
                __half2float(__ushort_as_half(packed.y)), __half2float(__ushort_as_half(packed.z)));
        }

        __device__ __forceinline__ float Rec709(const float3& v)
        { return v.x * 0.2126f + v.y * 0.7152f + v.z * 0.0722f; }

        __device__ __forceinline__ float LumaTimesTwo(const float3& v)
        { return v.z * 0.5f + v.x * 0.5f + v.y; }

        __device__ __forceinline__ float3 ReadLogical(
            cudaSurfaceObject_t source, int sampleWidth, int sampleHeight,
            int offsetX, int offsetY, int y, int x)
        {
            return ReadSource(source, sampleWidth, sampleHeight, y + offsetY, x + offsetX);
        }

        __device__ __forceinline__ void AccumulateDirection(
            float& dx, float& dy, float& length, float weight,
            float a, float b, float c, float d, float e)
        {
            constexpr float eps = 1.0e-12f;
            const float dc = d - c, cb = c - b;
            const float sx = d - b;
            const float lx = fmaxf(fabsf(dc), fabsf(cb));
            dx += sx * weight;
            const float rx = fminf(fmaxf(fabsf(sx) / fmaxf(lx, eps), 0.0f), 1.0f);
            length += rx * rx * weight;
            const float ec = e - c, ca = c - a;
            const float sy = e - a;
            const float ly = fmaxf(fabsf(ec), fabsf(ca));
            dy += sy * weight;
            const float ry = fminf(fmaxf(fabsf(sy) / fmaxf(ly, eps), 0.0f), 1.0f);
            length += ry * ry * weight;
        }

        __device__ __forceinline__ void Tap(
            float3& sum, float& weightSum, const float3& color,
            float ox, float oy, float dx, float dy, float lx, float ly,
            float lobe, float clip)
        {
            float vx = (ox * dx + oy * dy) * lx;
            float vy = (ox * -dy + oy * dx) * ly;
            const float distance2 = fminf(vx * vx + vy * vy, clip);
            float wb = (2.0f / 5.0f) * distance2 - 1.0f;
            float wa = lobe * distance2 - 1.0f;
            wb *= wb; wa *= wa;
            const float weight = ((25.0f / 16.0f) * wb - (25.0f / 16.0f - 1.0f)) * wa;
            sum.x += color.x * weight;
            sum.y += color.y * weight;
            sum.z += color.z * weight;
            weightSum += weight;
        }

        struct EasuValue
        {
            float3 color;
            float3 edgeBasis;
            float dx;
            float dy;
            float length;
            float range;
        };

        __device__ __forceinline__ EasuValue Easu(
            cudaSurfaceObject_t source, int logicalWidth, int logicalHeight,
            int sampleWidth, int sampleHeight, int offsetX, int offsetY,
            int outputX, int outputY, float edgeGateGain)
        {
            // This runtime is exact-2x by contract. Expressing the invariant
            // directly avoids two per-output FP32 divisions in the hottest
            // non-Tensor-Core kernel.
            (void)logicalWidth;
            (void)logicalHeight;
            const float sourceX=fmaf(float(outputX),0.5f,-0.25f);
            const float sourceY=fmaf(float(outputY),0.5f,-0.25f);
            const int fx = int(floorf(sourceX)), fy = int(floorf(sourceY));
            const float px = sourceX - fx, py = sourceY - fy;
            const float3 b = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy - 1,fx + 0);
            const float3 c = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy - 1,fx + 1);
            const float3 e = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 0,fx - 1);
            const float3 f = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 0,fx + 0);
            const float3 g = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 0,fx + 1);
            const float3 h = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 0,fx + 2);
            const float3 i = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 1,fx - 1);
            const float3 j = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 1,fx + 0);
            const float3 k = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 1,fx + 1);
            const float3 l = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 1,fx + 2);
            const float3 n = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 2,fx + 0);
            const float3 o = ReadLogical(source,sampleWidth,sampleHeight,offsetX,offsetY,fy + 2,fx + 1);
            const float bl=LumaTimesTwo(b), cl=LumaTimesTwo(c), el=LumaTimesTwo(e), fl=LumaTimesTwo(f), gl=LumaTimesTwo(g), hl=LumaTimesTwo(h);
            const float il=LumaTimesTwo(i), jl=LumaTimesTwo(j), kl=LumaTimesTwo(k), ll=LumaTimesTwo(l), nl=LumaTimesTwo(n), ol=LumaTimesTwo(o);
            float dx=0.0f, dy=0.0f, length=0.0f;
            const float ws=(1.0f-px)*(1.0f-py), wt=px*(1.0f-py), wu=(1.0f-px)*py, wv=px*py;
            AccumulateDirection(dx,dy,length,ws,bl,el,fl,gl,jl);
            AccumulateDirection(dx,dy,length,wt,cl,fl,gl,hl,kl);
            AccumulateDirection(dx,dy,length,wu,fl,il,jl,kl,nl);
            AccumulateDirection(dx,dy,length,wv,gl,jl,kl,ll,ol);
            const float dir2=dx*dx+dy*dy;
            const bool zero=dir2 < (1.0f/32768.0f);
            const float inv=zero ? 1.0f : rsqrtf(fmaxf(dir2,1.0e-12f));
            dx=(zero?1.0f:dx)*inv; dy=(zero?0.0f:dy)*inv;
            length*=0.5f; length*=length;
            const float rangeRaw=fmaxf(fmaxf(fl,gl),fmaxf(jl,kl))-fminf(fminf(fl,gl),fminf(jl,kl));
            const float metaLength=fminf(1.0f,fmaxf(0.0f,length));
            const float metaRange=fminf(1.0f,fmaxf(0.0f,rangeRaw*0.5f));
            const float maxDir=fmaxf(fabsf(dx),fabsf(dy));
            const float stretch=(dx*dx+dy*dy)/fmaxf(maxDir,1.0e-12f);
            const float lx=1.0f+(stretch-1.0f)*length, ly=1.0f-0.5f*length;
            const float lobe=0.5f+((0.25f-0.04f)-0.5f)*length, clip=1.0f/fmaxf(lobe,1.0e-12f);
            const float3 minimum=make_float3(fminf(fminf(f.x,g.x),fminf(j.x,k.x)),fminf(fminf(f.y,g.y),fminf(j.y,k.y)),fminf(fminf(f.z,g.z),fminf(j.z,k.z)));
            const float3 maximum=make_float3(fmaxf(fmaxf(f.x,g.x),fmaxf(j.x,k.x)),fmaxf(fmaxf(f.y,g.y),fmaxf(j.y,k.y)),fmaxf(fmaxf(f.z,g.z),fmaxf(j.z,k.z)));
            float3 sum=make_float3(0,0,0); float weightSum=0;
            Tap(sum,weightSum,b,-px,-1.0f-py,dx,dy,lx,ly,lobe,clip); Tap(sum,weightSum,c,1.0f-px,-1.0f-py,dx,dy,lx,ly,lobe,clip);
            Tap(sum,weightSum,i,-1.0f-px,1.0f-py,dx,dy,lx,ly,lobe,clip); Tap(sum,weightSum,j,-px,1.0f-py,dx,dy,lx,ly,lobe,clip);
            Tap(sum,weightSum,f,-px,-py,dx,dy,lx,ly,lobe,clip); Tap(sum,weightSum,e,-1.0f-px,-py,dx,dy,lx,ly,lobe,clip);
            Tap(sum,weightSum,k,1.0f-px,1.0f-py,dx,dy,lx,ly,lobe,clip); Tap(sum,weightSum,l,2.0f-px,1.0f-py,dx,dy,lx,ly,lobe,clip);
            Tap(sum,weightSum,h,2.0f-px,-py,dx,dy,lx,ly,lobe,clip); Tap(sum,weightSum,g,1.0f-px,-py,dx,dy,lx,ly,lobe,clip);
            Tap(sum,weightSum,o,1.0f-px,2.0f-py,dx,dy,lx,ly,lobe,clip); Tap(sum,weightSum,n,-px,2.0f-py,dx,dy,lx,ly,lobe,clip);
            const float invWeight=1.0f/fmaxf(weightSum,1.0e-12f);
            sum.x=fminf(1.0f,fmaxf(0.0f,fminf(maximum.x,fmaxf(minimum.x,sum.x*invWeight))));
            sum.y=fminf(1.0f,fmaxf(0.0f,fminf(maximum.y,fmaxf(minimum.y,sum.y*invWeight))));
            sum.z=fminf(1.0f,fmaxf(0.0f,fminf(maximum.z,fmaxf(minimum.z,sum.z*invWeight))));
            const float gate = fminf(
                1.0f,
                fmaxf(0.0f, metaLength * metaRange * fmaxf(edgeGateGain, 0.0f)));
            const float oneMinusX = 1.0f - px;
            const float oneMinusY = 1.0f - py;
            const float3 gradX = make_float3(
                (oneMinusY * (g.x - f.x) + py * (k.x - j.x)) * 0.5f,
                (oneMinusY * (g.y - f.y) + py * (k.y - j.y)) * 0.5f,
                (oneMinusY * (g.z - f.z) + py * (k.z - j.z)) * 0.5f);
            const float3 gradY = make_float3(
                (oneMinusX * (j.x - f.x) + px * (k.x - g.x)) * 0.5f,
                (oneMinusX * (j.y - f.y) + px * (k.y - g.y)) * 0.5f,
                (oneMinusX * (j.z - f.z) + px * (k.z - g.z)) * 0.5f);
            const float3 edgeBasis = make_float3(
                (gradX.x * dx + gradY.x * dy) * gate,
                (gradX.y * dx + gradY.y * dy) * gate,
                (gradX.z * dx + gradY.z * dy) * gate);
            return {sum,edgeBasis,dx,dy,metaLength,metaRange};
        }

        template <typename PackedT>
        __device__ __forceinline__ void StorePacked(
            PackedT* pointer, float value, float activationReciprocalScale);

        template <>
        __device__ __forceinline__ void StorePacked<__nv_fp8_e4m3>(
            __nv_fp8_e4m3* pointer, float value,
            float activationReciprocalScale)
        {
            *pointer = __nv_fp8_e4m3(value * activationReciprocalScale);
        }

        template <>
        __device__ __forceinline__ void StorePacked<Half>(
            Half* pointer, float value, float)
        {
            *pointer = Half(value);
        }

        struct EasuHalf3
        {
            __half x;
            __half y;
            __half z;
        };

        struct EasuHalfValue
        {
            EasuHalf3 color;
            EasuHalf3 edgeBasis;
            __half dx;
            __half dy;
            __half length;
            __half range;
        };

        template <typename PackedT>
        __device__ __forceinline__ void StorePackedHalf(
            PackedT* pointer, __half value,
            float activationReciprocalScale);

        template <>
        __device__ __forceinline__ void StorePackedHalf<__nv_fp8_e4m3>(
            __nv_fp8_e4m3* pointer, __half value,
            float activationReciprocalScale)
        {
            *pointer=__nv_fp8_e4m3(
                __half2float(value)*activationReciprocalScale);
        }

        template <>
        __device__ __forceinline__ void StorePackedHalf<Half>(
            Half* pointer, __half value, float)
        {
            *reinterpret_cast<unsigned short*>(pointer)=
                __half_as_ushort(value);
        }

        struct EasuHalf3Pair
        {
            __half2 x;
            __half2 y;
            __half2 z;
        };

        __device__ __forceinline__ __half EasuHalf(float value)
        {
            return __float2half_rn(value);
        }

        __device__ __forceinline__ __half2 EasuHalf2(__half low, __half high)
        {
            return __halves2half2(low, high);
        }

        __device__ __forceinline__ __half2 EasuHalf2(float low, float high)
        {
            return EasuHalf2(EasuHalf(low), EasuHalf(high));
        }

        __device__ __forceinline__ __half2 EasuHalf2Broadcast(__half value)
        {
            return __halves2half2(value, value);
        }

        __device__ __forceinline__ __half EasuHalfSaturate(__half value)
        {
            return __hmin(__hmax(value, EasuHalf(0.0f)), EasuHalf(1.0f));
        }

        __device__ __forceinline__ __half2 EasuHalf2Saturate(__half2 value)
        {
            return __hmin2(
                __hmax2(value, EasuHalf2(0.0f, 0.0f)),
                EasuHalf2(1.0f, 1.0f));
        }

        __device__ __forceinline__ EasuHalf3 EasuLoadHalf3(const ushort4& value)
        {
            return {
                __ushort_as_half(value.x),
                __ushort_as_half(value.y),
                __ushort_as_half(value.z)};
        }

        template <int TileWidth>
        __device__ __forceinline__ EasuHalf3 EasuReadTile(
            const ushort4* tile, int y, int x)
        {
            return EasuLoadHalf3(tile[y * TileWidth + x]);
        }

        __device__ __forceinline__ __half EasuLumaHalf(const EasuHalf3& value)
        {
            // Match the packed FP16 FidelityFX path: B*0.5 + (R*0.5 + G).
            return __hfma(
                EasuHalf(0.5f), value.z,
                __hfma(EasuHalf(0.5f), value.x, value.y));
        }

        __device__ __forceinline__ __half EasuRec709Half(const EasuHalf3& value)
        {
            return __hfma(
                EasuHalf(0.2126f), value.x,
                __hfma(
                    EasuHalf(0.7152f), value.y,
                    __hmul(EasuHalf(0.0722f), value.z)));
        }

        __device__ __forceinline__ EasuHalf3Pair EasuPair(
            const EasuHalf3& low, const EasuHalf3& high)
        {
            return {
                EasuHalf2(low.x, high.x),
                EasuHalf2(low.y, high.y),
                EasuHalf2(low.z, high.z)};
        }

        __device__ __forceinline__ void EasuAccumulateDirectionHalf2(
            __half2& directionX, __half2& directionY, __half2& length,
            __half2 weight, __half2 a, __half2 b, __half2 c,
            __half2 d, __half2 e)
        {
            // A half reciprocal over a zero gradient would produce an infinity
            // before the zero numerator is applied. 1/32768 is also the
            // canonical EASU zero-direction threshold and keeps this path
            // finite without changing meaningful gradients.
            const __half2 epsilon = EasuHalf2(
                1.0f / 32768.0f, 1.0f / 32768.0f);
            const __half2 dc = __hsub2(d, c);
            const __half2 cb = __hsub2(c, b);
            const __half2 directionDeltaX = __hsub2(d, b);
            const __half2 reciprocalX = h2rcp(__hmax2(
                __hmax2(__habs2(dc), __habs2(cb)), epsilon));
            directionX = __hfma2(directionDeltaX, weight, directionX);
            __half2 ratioX = EasuHalf2Saturate(
                __hmul2(__habs2(directionDeltaX), reciprocalX));
            length = __hfma2(__hmul2(ratioX, ratioX), weight, length);

            const __half2 ec = __hsub2(e, c);
            const __half2 ca = __hsub2(c, a);
            const __half2 directionDeltaY = __hsub2(e, a);
            const __half2 reciprocalY = h2rcp(__hmax2(
                __hmax2(__habs2(ec), __habs2(ca)), epsilon));
            directionY = __hfma2(directionDeltaY, weight, directionY);
            __half2 ratioY = EasuHalf2Saturate(
                __hmul2(__habs2(directionDeltaY), reciprocalY));
            length = __hfma2(__hmul2(ratioY, ratioY), weight, length);
        }

        __device__ __forceinline__ void EasuAccumulateTapHalf2(
            EasuHalf3Pair& color, __half2& weightSum,
            __half2 offsetX, __half2 offsetY,
            __half directionX, __half directionY,
            __half lengthX, __half lengthY,
            __half lobe, __half clip, const EasuHalf3Pair& sample)
        {
            const __half2 dx = EasuHalf2Broadcast(directionX);
            const __half2 dy = EasuHalf2Broadcast(directionY);
            __half2 vectorX = __hfma2(offsetX, dx, __hmul2(offsetY, dy));
            __half2 vectorY = __hfma2(
                offsetX, EasuHalf2Broadcast(__hneg(directionY)),
                __hmul2(offsetY, dx));
            vectorX = __hmul2(vectorX, EasuHalf2Broadcast(lengthX));
            vectorY = __hmul2(vectorY, EasuHalf2Broadcast(lengthY));
            __half2 distanceSquared = __hfma2(
                vectorX, vectorX, __hmul2(vectorY, vectorY));
            distanceSquared = __hmin2(
                distanceSquared, EasuHalf2Broadcast(clip));
            __half2 base = __hfma2(
                EasuHalf2(2.0f / 5.0f, 2.0f / 5.0f),
                distanceSquared, EasuHalf2(-1.0f, -1.0f));
            __half2 window = __hfma2(
                EasuHalf2Broadcast(lobe), distanceSquared,
                EasuHalf2(-1.0f, -1.0f));
            base = __hmul2(base, base);
            window = __hmul2(window, window);
            base = __hfma2(
                EasuHalf2(25.0f / 16.0f, 25.0f / 16.0f), base,
                EasuHalf2(-(25.0f / 16.0f - 1.0f),
                          -(25.0f / 16.0f - 1.0f)));
            const __half2 weight = __hmul2(base, window);
            color.x = __hfma2(sample.x, weight, color.x);
            color.y = __hfma2(sample.y, weight, color.y);
            color.z = __hfma2(sample.z, weight, color.z);
            weightSum = __hadd2(weightSum, weight);
        }

        __device__ __forceinline__ __half EasuHalfMin4(
            __half a, __half b, __half c, __half d)
        {
            return __hmin(__hmin(a, b), __hmin(c, d));
        }

        __device__ __forceinline__ __half EasuHalfMax4(
            __half a, __half b, __half c, __half d)
        {
            return __hmax(__hmax(a, b), __hmax(c, d));
        }

        __device__ __forceinline__ __half EasuGradientHalf(
            __half nearDelta, __half farDelta,
            __half nearWeight, __half farWeight)
        {
            return __hmul(
                __hfma(farDelta, farWeight,
                    __hmul(nearDelta, nearWeight)),
                EasuHalf(0.5f));
        }

        template <int TileWidth>
        __device__ __forceinline__ EasuHalfValue EasuTiledHalf(
            const ushort4* tile, int localX, int localY,
            __half edgeGateGain)
        {
            // For exact 2x scaling the fractional source position has only
            // four phases. Resolve the integer footprint with shifts and make
            // the .25/.75 fractions immediate FP16 constants.
            const int footprintX = 1 + ((localX + 1) >> 1);
            const int footprintY = 1 + ((localY + 1) >> 1);
            const __half positionX = EasuHalf((localX & 1) ? 0.25f : 0.75f);
            const __half positionY = EasuHalf((localY & 1) ? 0.25f : 0.75f);
            const __half oneMinusX = __hsub(EasuHalf(1.0f), positionX);
            const __half oneMinusY = __hsub(EasuHalf(1.0f), positionY);

            const EasuHalf3 b = EasuReadTile<TileWidth>(tile, footprintY - 1, footprintX + 0);
            const EasuHalf3 c = EasuReadTile<TileWidth>(tile, footprintY - 1, footprintX + 1);
            const EasuHalf3 e = EasuReadTile<TileWidth>(tile, footprintY + 0, footprintX - 1);
            const EasuHalf3 f = EasuReadTile<TileWidth>(tile, footprintY + 0, footprintX + 0);
            const EasuHalf3 g = EasuReadTile<TileWidth>(tile, footprintY + 0, footprintX + 1);
            const EasuHalf3 h = EasuReadTile<TileWidth>(tile, footprintY + 0, footprintX + 2);
            const EasuHalf3 i = EasuReadTile<TileWidth>(tile, footprintY + 1, footprintX - 1);
            const EasuHalf3 j = EasuReadTile<TileWidth>(tile, footprintY + 1, footprintX + 0);
            const EasuHalf3 k = EasuReadTile<TileWidth>(tile, footprintY + 1, footprintX + 1);
            const EasuHalf3 l = EasuReadTile<TileWidth>(tile, footprintY + 1, footprintX + 2);
            const EasuHalf3 n = EasuReadTile<TileWidth>(tile, footprintY + 2, footprintX + 0);
            const EasuHalf3 o = EasuReadTile<TileWidth>(tile, footprintY + 2, footprintX + 1);

            const __half bl = EasuLumaHalf(b);
            const __half cl = EasuLumaHalf(c);
            const __half el = EasuLumaHalf(e);
            const __half fl = EasuLumaHalf(f);
            const __half gl = EasuLumaHalf(g);
            const __half hl = EasuLumaHalf(h);
            const __half il = EasuLumaHalf(i);
            const __half jl = EasuLumaHalf(j);
            const __half kl = EasuLumaHalf(k);
            const __half ll = EasuLumaHalf(l);
            const __half nl = EasuLumaHalf(n);
            const __half ol = EasuLumaHalf(o);

            const __half2 weightTop = EasuHalf2(
                __hmul(oneMinusX, oneMinusY),
                __hmul(positionX, oneMinusY));
            const __half2 weightBottom = EasuHalf2(
                __hmul(oneMinusX, positionY),
                __hmul(positionX, positionY));
            __half2 directionPairX = EasuHalf2(0.0f, 0.0f);
            __half2 directionPairY = EasuHalf2(0.0f, 0.0f);
            __half2 lengthPair = EasuHalf2(0.0f, 0.0f);
            EasuAccumulateDirectionHalf2(
                directionPairX, directionPairY, lengthPair, weightTop,
                EasuHalf2(bl, cl), EasuHalf2(el, fl), EasuHalf2(fl, gl),
                EasuHalf2(gl, hl), EasuHalf2(jl, kl));
            EasuAccumulateDirectionHalf2(
                directionPairX, directionPairY, lengthPair, weightBottom,
                EasuHalf2(fl, gl), EasuHalf2(il, jl), EasuHalf2(jl, kl),
                EasuHalf2(kl, ll), EasuHalf2(nl, ol));

            __half directionX = __hadd(
                __low2half(directionPairX), __high2half(directionPairX));
            __half directionY = __hadd(
                __low2half(directionPairY), __high2half(directionPairY));
            __half length = __hadd(
                __low2half(lengthPair), __high2half(lengthPair));
            __half directionSquared = __hfma(
                directionX, directionX, __hmul(directionY, directionY));
            const bool zeroDirection =
                __half2float(directionSquared) < (1.0f / 32768.0f);
            const __half inverseDirection = zeroDirection
                ? EasuHalf(1.0f)
                : hrsqrt(directionSquared);
            directionX = zeroDirection
                ? EasuHalf(1.0f)
                : __hmul(directionX, inverseDirection);
            directionY = zeroDirection
                ? EasuHalf(0.0f)
                : __hmul(directionY, inverseDirection);
            length = __hmul(length, EasuHalf(0.5f));
            length = EasuHalfSaturate(__hmul(length, length));

            const __half rangeRaw = __hsub(
                EasuHalfMax4(fl, gl, jl, kl),
                EasuHalfMin4(fl, gl, jl, kl));
            const __half range = EasuHalfSaturate(
                __hmul(rangeRaw, EasuHalf(0.5f)));
            const __half maxDirection = __hmax(
                __habs(directionX), __habs(directionY));
            const __half stretch = __hmul(
                __hfma(directionX, directionX,
                    __hmul(directionY, directionY)),
                hrcp(__hmax(maxDirection, EasuHalf(1.0f / 32768.0f))));
            const __half lengthX = __hfma(
                __hsub(stretch, EasuHalf(1.0f)), length, EasuHalf(1.0f));
            const __half lengthY = __hfma(
                EasuHalf(-0.5f), length, EasuHalf(1.0f));
            const __half lobe = __hfma(
                EasuHalf((0.25f - 0.04f) - 0.5f),
                length, EasuHalf(0.5f));
            const __half clip = hrcp(lobe);

            EasuHalf3 minimum{
                EasuHalfMin4(f.x, g.x, j.x, k.x),
                EasuHalfMin4(f.y, g.y, j.y, k.y),
                EasuHalfMin4(f.z, g.z, j.z, k.z)};
            EasuHalf3 maximum{
                EasuHalfMax4(f.x, g.x, j.x, k.x),
                EasuHalfMax4(f.y, g.y, j.y, k.y),
                EasuHalfMax4(f.z, g.z, j.z, k.z)};
            EasuHalf3Pair color{
                EasuHalf2(0.0f, 0.0f),
                EasuHalf2(0.0f, 0.0f),
                EasuHalf2(0.0f, 0.0f)};
            __half2 weightSum = EasuHalf2(0.0f, 0.0f);
            const __half2 positionXX = EasuHalf2Broadcast(positionX);
            const __half2 positionYY = EasuHalf2Broadcast(positionY);
            EasuAccumulateTapHalf2(
                color, weightSum,
                __hsub2(EasuHalf2(0.0f, 1.0f), positionXX),
                __hsub2(EasuHalf2(-1.0f, -1.0f), positionYY),
                directionX, directionY, lengthX, lengthY, lobe, clip,
                EasuPair(b, c));
            EasuAccumulateTapHalf2(
                color, weightSum,
                __hsub2(EasuHalf2(-1.0f, 0.0f), positionXX),
                __hsub2(EasuHalf2(1.0f, 1.0f), positionYY),
                directionX, directionY, lengthX, lengthY, lobe, clip,
                EasuPair(i, j));
            EasuAccumulateTapHalf2(
                color, weightSum,
                __hsub2(EasuHalf2(0.0f, -1.0f), positionXX),
                __hsub2(EasuHalf2(0.0f, 0.0f), positionYY),
                directionX, directionY, lengthX, lengthY, lobe, clip,
                EasuPair(f, e));
            EasuAccumulateTapHalf2(
                color, weightSum,
                __hsub2(EasuHalf2(1.0f, 2.0f), positionXX),
                __hsub2(EasuHalf2(1.0f, 1.0f), positionYY),
                directionX, directionY, lengthX, lengthY, lobe, clip,
                EasuPair(k, l));
            EasuAccumulateTapHalf2(
                color, weightSum,
                __hsub2(EasuHalf2(2.0f, 1.0f), positionXX),
                __hsub2(EasuHalf2(0.0f, 0.0f), positionYY),
                directionX, directionY, lengthX, lengthY, lobe, clip,
                EasuPair(h, g));
            EasuAccumulateTapHalf2(
                color, weightSum,
                __hsub2(EasuHalf2(1.0f, 0.0f), positionXX),
                __hsub2(EasuHalf2(2.0f, 2.0f), positionYY),
                directionX, directionY, lengthX, lengthY, lobe, clip,
                EasuPair(o, n));

            const __half combinedWeight = __hadd(
                __low2half(weightSum), __high2half(weightSum));
            const __half inverseWeight = hrcp(__hmax(
                combinedWeight, EasuHalf(1.0f / 32768.0f)));
            EasuHalf3 resolved{
                __hmul(__hadd(__low2half(color.x), __high2half(color.x)),
                    inverseWeight),
                __hmul(__hadd(__low2half(color.y), __high2half(color.y)),
                    inverseWeight),
                __hmul(__hadd(__low2half(color.z), __high2half(color.z)),
                    inverseWeight)};
            resolved.x = __hmin(maximum.x, __hmax(minimum.x, resolved.x));
            resolved.y = __hmin(maximum.y, __hmax(minimum.y, resolved.y));
            resolved.z = __hmin(maximum.z, __hmax(minimum.z, resolved.z));
            resolved.x = EasuHalfSaturate(resolved.x);
            resolved.y = EasuHalfSaturate(resolved.y);
            resolved.z = EasuHalfSaturate(resolved.z);

            const __half gate = EasuHalfSaturate(__hmul(
                __hmul(length, range), edgeGateGain));
            const EasuHalf3 gradientX{
                EasuGradientHalf(__hsub(g.x, f.x), __hsub(k.x, j.x),
                    oneMinusY, positionY),
                EasuGradientHalf(__hsub(g.y, f.y), __hsub(k.y, j.y),
                    oneMinusY, positionY),
                EasuGradientHalf(__hsub(g.z, f.z), __hsub(k.z, j.z),
                    oneMinusY, positionY)};
            const EasuHalf3 gradientY{
                EasuGradientHalf(__hsub(j.x, f.x), __hsub(k.x, g.x),
                    oneMinusX, positionX),
                EasuGradientHalf(__hsub(j.y, f.y), __hsub(k.y, g.y),
                    oneMinusX, positionX),
                EasuGradientHalf(__hsub(j.z, f.z), __hsub(k.z, g.z),
                    oneMinusX, positionX)};
            const EasuHalf3 edgeBasis{
                __hmul(__hfma(gradientX.x, directionX,
                    __hmul(gradientY.x, directionY)), gate),
                __hmul(__hfma(gradientX.y, directionX,
                    __hmul(gradientY.y, directionY)), gate),
                __hmul(__hfma(gradientX.z, directionX,
                    __hmul(gradientY.z, directionY)), gate)};

            return {
                resolved,edgeBasis,directionX,directionY,length,range};
        }

        template <int TileWidth, typename PackedT>
        __device__ __forceinline__ void EasuPrepareTiledHalfPixel(
            const ushort4* sourceTile, int localX, int localY,
            int outputX, int outputY, int paddedWidth, int paddedHeight,
            int outputWidth, int outputHeight, ushort4* edge, PackedT* packed,
            float activationReciprocalScale, __half edgeGateGain,
            cudaSurfaceObject_t output)
        {
            const int paddedOutputWidth = paddedWidth * 2;
            const int paddedOutputHeight = paddedHeight * 2;
            if (outputX >= paddedOutputWidth || outputY >= paddedOutputHeight)
                return;

            const EasuHalfValue value = EasuTiledHalf<TileWidth>(
                sourceTile, localX, localY, edgeGateGain);
            constexpr int trim = 16;
            const int coreX = outputX - trim;
            const int coreY = outputY - trim;
            const bool core = coreX >= 0 && coreX < outputWidth &&
                coreY >= 0 && coreY < outputHeight;
            if (core)
            {
                const ushort4 baseValue = make_ushort4(
                    __half_as_ushort(value.color.x),
                    __half_as_ushort(value.color.y),
                    __half_as_ushort(value.color.z),
                    __half_as_ushort(__float2half_rn(1.0f)));
                edge[size_t(coreY) * outputWidth + coreX] = make_ushort4(
                    __half_as_ushort(value.edgeBasis.x),
                    __half_as_ushort(value.edgeBasis.y),
                    __half_as_ushort(value.edgeBasis.z),0);
                surf2Dwrite(
                    baseValue, output, coreX * int(sizeof(ushort4)), coreY);
            }

            const int packedY = outputY >> 2;
            const int packedX = outputX >> 2;
            const int phaseFirst = ((outputY & 1) << 1) | (outputX & 1);
            const int phaseSecond = (((outputY >> 1) & 1) << 1) |
                ((outputX >> 1) & 1);
            const int phaseNested = phaseFirst * 4 + phaseSecond;
            const int packedWidth = paddedWidth / 2;
            const size_t packedBase =
                (size_t(packedY) * packedWidth + packedX) * 80;
            if ((localY & 3) < 2 && (localX & 3) < 2)
            {
                const int sourceTileY = 2 + ((localY >> 2) << 1) +
                    (localY & 1);
                const int sourceTileX = 2 + ((localX >> 2) << 1) +
                    (localX & 1);
                const EasuHalf3 sourceValue = EasuReadTile<TileWidth>(
                    sourceTile, sourceTileY, sourceTileX);
                const int sourcePhase = ((outputY & 1) << 1) |
                    (outputX & 1);
                StorePackedHalf(
                    packed + packedBase + 0 * 4 + sourcePhase,
                    sourceValue.x,activationReciprocalScale);
                StorePackedHalf(
                    packed + packedBase + 1 * 4 + sourcePhase,
                    sourceValue.y,activationReciprocalScale);
                StorePackedHalf(
                    packed + packedBase + 2 * 4 + sourcePhase,
                    sourceValue.z,activationReciprocalScale);
                StorePackedHalf(
                    packed + packedBase + 3 * 4 + sourcePhase,
                    EasuRec709Half(sourceValue),
                    activationReciprocalScale);
            }
            const __half metadata[4] = {
                value.dx, value.dy, value.length, value.range};
#pragma unroll
            for (int metadataIndex = 0; metadataIndex < 4; ++metadataIndex)
            {
                StorePackedHalf(
                    packed + packedBase + 16 + metadataIndex * 16 +
                        phaseNested,
                    metadata[metadataIndex], activationReciprocalScale);
            }
        }

        template <int OutputTileWidth, int OutputTileHeight>
        __device__ __forceinline__ void EasuLoadSourceTile(
            ushort4* sourceTile, cudaSurfaceObject_t source,
            int sourceWidth, int sourceHeight, int padLeft, int padTop)
        {
            constexpr int tileWidth = OutputTileWidth / 2 + 4;
            constexpr int tileHeight = OutputTileHeight / 2 + 4;
            const int threadIndex =
                int(threadIdx.y) * OutputTileWidth + int(threadIdx.x);
            constexpr int threadCount = OutputTileWidth * OutputTileHeight;
            for (int tileIndex = threadIndex;
                tileIndex < tileWidth * tileHeight;
                tileIndex += threadCount)
            {
                const int tileX = tileIndex % tileWidth;
                const int tileY = tileIndex / tileWidth;
                const int sourceX =
                    int(blockIdx.x) * (OutputTileWidth / 2) - 2 -
                    padLeft + tileX;
                const int sourceY =
                    int(blockIdx.y) * (OutputTileHeight / 2) - 2 -
                    padTop + tileY;
                const int clampedX = ClampInt(sourceX, 0, sourceWidth - 1);
                const int clampedY = ClampInt(sourceY, 0, sourceHeight - 1);
                sourceTile[tileIndex] = surf2Dread<ushort4>(
                    source, clampedX * int(sizeof(ushort4)), clampedY);
            }
        }

        template <int OutputTileWidth, int OutputTileHeight, typename PackedT>
        __global__ void EasuPrepareTiledHalfKernel(
            cudaSurfaceObject_t source, int sourceWidth, int sourceHeight,
            int paddedWidth, int paddedHeight, int padLeft, int padTop,
            int outputWidth, int outputHeight, ushort4* edge, PackedT* packed,
            float activationReciprocalScale, __half edgeGateGain,
            cudaSurfaceObject_t output)
        {
            constexpr int tileWidth = OutputTileWidth / 2 + 4;
            constexpr int tileHeight = OutputTileHeight / 2 + 4;
            __shared__ ushort4 sourceTile[tileWidth * tileHeight];
            EasuLoadSourceTile<OutputTileWidth,OutputTileHeight>(
                sourceTile, source, sourceWidth, sourceHeight,
                padLeft, padTop);
            __syncthreads();
            const int localX = int(threadIdx.x);
            const int localY = int(threadIdx.y);
            EasuPrepareTiledHalfPixel<tileWidth>(
                sourceTile, localX, localY,
                int(blockIdx.x) * OutputTileWidth + localX,
                int(blockIdx.y) * OutputTileHeight + localY,
                paddedWidth, paddedHeight, outputWidth, outputHeight,
                edge, packed, activationReciprocalScale, edgeGateGain, output);
        }

        template <typename PackedT>
        __global__ void EasuPrepareReferenceKernel(
            cudaSurfaceObject_t source, int sourceWidth, int sourceHeight,
            int paddedWidth, int paddedHeight, int padLeft, int padTop,
            int outputWidth,int outputHeight,ushort4* edge,PackedT* packed,
            float activationReciprocalScale,float edgeGateGain,
            cudaSurfaceObject_t output)
        {
            const int ox=int(blockIdx.x*blockDim.x+threadIdx.x), oy=int(blockIdx.y*blockDim.y+threadIdx.y);
            const int outW=paddedWidth*2, outH=paddedHeight*2;
            if (ox>=outW || oy>=outH) return;
            const EasuValue v=Easu(
                source,paddedWidth,paddedHeight,sourceWidth,sourceHeight,
                -padLeft,-padTop,ox,oy,edgeGateGain);
            constexpr int trim = 16;
            const int coreX = ox - trim;
            const int coreY = oy - trim;
            const bool core = coreX >= 0 && coreX < outputWidth &&
                coreY >= 0 && coreY < outputHeight;
            if (core)
            {
                const ushort4 baseValue=make_ushort4(
                    __half_as_ushort(__float2half_rn(v.color.x)),
                    __half_as_ushort(__float2half_rn(v.color.y)),
                    __half_as_ushort(__float2half_rn(v.color.z)),
                    __half_as_ushort(__float2half_rn(1.0f)));
                edge[size_t(coreY)*outputWidth+coreX]=make_ushort4(
                    __half_as_ushort(__float2half_rn(v.edgeBasis.x)),
                    __half_as_ushort(__float2half_rn(v.edgeBasis.y)),
                    __half_as_ushort(__float2half_rn(v.edgeBasis.z)),0);
                surf2Dwrite(
                    baseValue,output,coreX*int(sizeof(ushort4)),coreY);
            }

            const int py=oy>>2, px=ox>>2;
            const int phaseMetaFirst=((oy&1)<<1)|(ox&1);
            const int phaseMetaSecond=(((oy>>1)&1)<<1)|((ox>>1)&1);
            const int phaseNested=phaseMetaFirst*4+phaseMetaSecond;
            const int packedW=paddedWidth/2;
            const size_t packedBase=(size_t(py)*packedW+px)*80;
            const int sourcePhase=((oy&1)<<1)|(ox&1);
            if ((oy&3)<2 && (ox&3)<2)
            {
                const int sourceY=(py<<1)+(oy&1);
                const int sourceX=(px<<1)+(ox&1);
                const float3 src=ReadLogical(
                    source,sourceWidth,sourceHeight,-padLeft,-padTop,
                    sourceY,sourceX);
                StorePacked(packed+packedBase+0*4+sourcePhase,src.x,
                    activationReciprocalScale);
                StorePacked(packed+packedBase+1*4+sourcePhase,src.y,
                    activationReciprocalScale);
                StorePacked(packed+packedBase+2*4+sourcePhase,src.z,
                    activationReciprocalScale);
                StorePacked(packed+packedBase+3*4+sourcePhase,Rec709(src),
                    activationReciprocalScale);
            }
            const float metadata[4]={v.dx,v.dy,v.length,v.range};
#pragma unroll
            for (int m=0;m<4;++m)
                StorePacked(packed+packedBase+16+m*16+phaseNested,metadata[m],
                    activationReciprocalScale);
        }

        __global__ void ApplyProjectFp8EpilogueKernel(
            const float* accumulator, __nv_fp8_e4m3* skip,
            __nv_fp8_e4m3* unshuffled, int height, int width,
            float inputScale, float outputScale, const float* weightScale,
            const float* bias, const float* prelu)
        {
            constexpr int channels = 48;
            const size_t count=size_t(height)*width*channels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%channels);
            const size_t pixel=index/channels;
            const int x=int(pixel%width), y=int(pixel/width);
            float value=accumulator[index]*(inputScale*weightScale[c])+bias[c];
            if(value<0.0f) value*=prelu[c];
            const __nv_fp8_e4m3 quantized(value/outputScale);
            skip[index]=quantized;
            const int phase=(y&1)*2+(x&1);
            const size_t destination=(
                (size_t(y>>1)*(width>>1)+(x>>1))*channels+c)*4+phase;
            unshuffled[destination]=quantized;
        }

        __global__ void ApplyProjectHalfEpilogueKernel(
            const float* accumulator, Half* skip, Half* unshuffled,
            int height, int width, const float* bias, const float* prelu)
        {
            constexpr int channels = 48;
            const size_t count=size_t(height)*width*channels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%channels);
            const size_t pixel=index/channels;
            const int x=int(pixel%width), y=int(pixel/width);
            float value=accumulator[index]+bias[c];
            if(value<0.0f) value*=prelu[c];
            const Half quantized(value);
            skip[index]=quantized;
            const int phase=(y&1)*2+(x&1);
            const size_t destination=(
                (size_t(y>>1)*(width>>1)+(x>>1))*channels+c)*4+phase;
            unshuffled[destination]=quantized;
        }

        template <int Channels = 48>
        __global__ void PixelUnshuffle2Fp8Kernel(
            const __nv_fp8_e4m3* input,__nv_fp8_e4m3* output,
            int lowHeight,int lowWidth)
        {
            const size_t count=size_t(lowHeight)*lowWidth*Channels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%Channels);
            const size_t pixel=index/Channels;
            const int x=int(pixel%lowWidth),y=int(pixel/lowWidth);
            const int inputWidth=lowWidth*2;
            const size_t topLeft=(size_t(y*2)*inputWidth+x*2)*Channels+c;
            uchar4 packed{};
            packed.x=input[topLeft].__x;
            packed.y=input[topLeft+Channels].__x;
            packed.z=input[topLeft+size_t(inputWidth)*Channels].__x;
            packed.w=input[topLeft+size_t(inputWidth+1)*Channels].__x;
            reinterpret_cast<uchar4*>(output)[index]=packed;
        }

        template <int Channels = 48>
        __global__ void PixelUnshuffle2HalfKernel(
            const Half* input,Half* output,int lowHeight,int lowWidth)
        {
            const size_t count=size_t(lowHeight)*lowWidth*Channels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%Channels);
            const size_t pixel=index/Channels;
            const int x=int(pixel%lowWidth),y=int(pixel/lowWidth);
            const int inputWidth=lowWidth*2;
            const size_t topLeft=(size_t(y*2)*inputWidth+x*2)*Channels+c;
            const auto* inputBits=reinterpret_cast<const unsigned short*>(input);
            reinterpret_cast<ushort4*>(output)[index]=make_ushort4(
                inputBits[topLeft],inputBits[topLeft+Channels],
                inputBits[topLeft+size_t(inputWidth)*Channels],
                inputBits[topLeft+size_t(inputWidth+1)*Channels]);
        }

        template <int Channels = 32>
        __global__ void PixelUnshuffle2QuantizeFp8Kernel(
            const Half* input,__nv_fp8_e4m3* output,
            int lowHeight,int lowWidth,float reciprocalScale)
        {
            const size_t count=size_t(lowHeight)*lowWidth*Channels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%Channels);
            const size_t pixel=index/Channels;
            const int x=int(pixel%lowWidth),y=int(pixel/lowWidth);
            const int inputWidth=lowWidth*2;
            const size_t topLeft=(size_t(y*2)*inputWidth+x*2)*Channels+c;
            const float v0=float(input[topLeft])*reciprocalScale;
            const float v1=float(input[topLeft+Channels])*reciprocalScale;
            const float v2=float(input[topLeft+size_t(inputWidth)*Channels])*reciprocalScale;
            const float v3=float(input[topLeft+size_t(inputWidth+1)*Channels])*reciprocalScale;
            uchar4 packed{};
            packed.x=__nv_fp8_e4m3(v0).__x;
            packed.y=__nv_fp8_e4m3(v1).__x;
            packed.z=__nv_fp8_e4m3(v2).__x;
            packed.w=__nv_fp8_e4m3(v3).__x;
            reinterpret_cast<uchar4*>(output)[index]=packed;
        }

        __global__ void ApplyFp8EpilogueKernel(
            const float* accumulator, __nv_fp8_e4m3* output, size_t count, int channels,
            float inputScale, float outputScale, const float* weightScale,
            const float* bias, const float* prelu)
        {
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%channels);
            float value=accumulator[index]*(inputScale*weightScale[c])+bias[c];
            if(value<0.0f) value*=prelu[c];
            output[index]=__nv_fp8_e4m3(value/outputScale);
        }

        __global__ void ApplyHalfEpilogueKernel(
            const float* accumulator, Half* output, size_t count, int channels,
            const float* bias, const float* prelu)
        {
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%channels);
            float value=accumulator[index]+bias[c];
            if(value<0.0f) value*=prelu[c];
            output[index]=__float2half_rn(value);
        }

        __global__ void ApplyExpandFp8EpilogueKernel(
            const float* accumulator, const __nv_fp8_e4m3* skip,
            __nv_fp8_e4m3* output, int lowHeight, int lowWidth,
            float inputScale, float expandedScale, float skipScale,
            float outputScale, const float* weightScale,
            const float* bias, const float* prelu)
        {
            constexpr int expandedChannels=192, channels=48;
            const size_t count=size_t(lowHeight)*lowWidth*expandedChannels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int expandedChannel=int(index%expandedChannels);
            const int c=expandedChannel>>2, phase=expandedChannel&3;
            const size_t pixel=index/expandedChannels;
            const int x=int(pixel%lowWidth), y=int(pixel/lowWidth);
            float value=accumulator[index]*(inputScale*weightScale[expandedChannel])+
                bias[expandedChannel];
            if(value<0.0f) value*=prelu[expandedChannel];
            const __nv_fp8_e4m3 expanded(value/expandedScale);
            const int outputX=x*2+(phase&1), outputY=y*2+(phase>>1);
            const size_t destination=(
                size_t(outputY)*(lowWidth*2)+outputX)*channels+c;
            const float combined=float(expanded)*expandedScale+
                float(skip[destination])*skipScale;
            output[destination]=__nv_fp8_e4m3(combined/outputScale);
        }

        __global__ void ApplyExpandHalfEpilogueKernel(
            const float* accumulator, const Half* skip, Half* output,
            int lowHeight, int lowWidth, const float* bias, const float* prelu)
        {
            constexpr int expandedChannels=192, channels=48;
            const size_t count=size_t(lowHeight)*lowWidth*expandedChannels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int expandedChannel=int(index%expandedChannels);
            const int c=expandedChannel>>2, phase=expandedChannel&3;
            const size_t pixel=index/expandedChannels;
            const int x=int(pixel%lowWidth), y=int(pixel/lowWidth);
            float value=accumulator[index]+bias[expandedChannel];
            if(value<0.0f) value*=prelu[expandedChannel];
            const Half expanded(value);
            const int outputX=x*2+(phase&1), outputY=y*2+(phase>>1);
            const size_t destination=(
                size_t(outputY)*(lowWidth*2)+outputX)*channels+c;
            output[destination]=Half(float(expanded)+float(skip[destination]));
        }

        template <int Channels = 48, int outputChannels = Channels>
        __global__ void PixelShuffle2SkipAddFp8Kernel(
            const __nv_fp8_e4m3* expanded,const __nv_fp8_e4m3* skip,
            __nv_fp8_e4m3* output,int lowHeight,int lowWidth,
            float expandedScale,float skipScale,
            float outputReciprocalScale)
        {
            const size_t count=size_t(lowHeight)*lowWidth*outputChannels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%outputChannels);
            const size_t pixel=index/outputChannels;
            const int x=int(pixel%lowWidth),y=int(pixel/lowWidth);
            const uchar4 packed=c<Channels ? reinterpret_cast<const uchar4*>(expanded)[pixel*Channels+c] : make_uchar4(0,0,0,0);
            const unsigned char codes[4]={packed.x,packed.y,packed.z,packed.w};
#pragma unroll
            for(int phase=0;phase<4;++phase)
            {
                __nv_fp8_e4m3 expandedValue;
                expandedValue.__x=codes[phase];
                const int outputX=x*2+(phase&1);
                const int outputY=y*2+(phase>>1);
                const size_t outputPixel=size_t(outputY)*(lowWidth*2)+outputX;
                const size_t destination=outputPixel*outputChannels+c;
                const float combined=c<Channels ? float(expandedValue)*expandedScale+
                    float(skip[outputPixel*Channels+c])*skipScale : 0.0f;
                output[destination]=__nv_fp8_e4m3(
                    combined*outputReciprocalScale);
            }
        }

        template <int Channels = 48>
        __global__ void PixelShuffle2SkipAddHalfKernel(
            const Half* expanded,const Half* skip,Half* output,
            int lowHeight,int lowWidth)
        {
            const size_t count=size_t(lowHeight)*lowWidth*Channels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%Channels);
            const size_t pixel=index/Channels;
            const int x=int(pixel%lowWidth),y=int(pixel/lowWidth);
            const ushort4 packed=
                reinterpret_cast<const ushort4*>(expanded)[index];
            const unsigned short codes[4]={
                packed.x,packed.y,packed.z,packed.w};
            const auto* skipBits=reinterpret_cast<const unsigned short*>(skip);
            auto* outputBits=reinterpret_cast<unsigned short*>(output);
#pragma unroll
            for(int phase=0;phase<4;++phase)
            {
                const int outputX=x*2+(phase&1);
                const int outputY=y*2+(phase>>1);
                const size_t destination=(
                    size_t(outputY)*(lowWidth*2)+outputX)*Channels+c;
                const float combined=
                    __half2float(__ushort_as_half(codes[phase]))+
                    __half2float(__ushort_as_half(skipBits[destination]));
                outputBits[destination]=
                    __half_as_ushort(__float2half_rn(combined));
            }
        }

        template <int Channels = 32>
        __global__ void PixelShuffle2DequantizeSkipAddHalfKernel(
            const __nv_fp8_e4m3* expanded,const Half* skip,Half* output,
            int lowHeight,int lowWidth,float expandedScale)
        {
            const size_t count=size_t(lowHeight)*lowWidth*Channels;
            const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(index>=count) return;
            const int c=int(index%Channels);
            const size_t pixel=index/Channels;
            const int x=int(pixel%lowWidth),y=int(pixel/lowWidth);
            const uchar4 packed=
                reinterpret_cast<const uchar4*>(expanded)[index];
            const unsigned char codes[4]={
                packed.x,packed.y,packed.z,packed.w};
            const auto* skipBits=reinterpret_cast<const unsigned short*>(skip);
            auto* outputBits=reinterpret_cast<unsigned short*>(output);
#pragma unroll
            for(int phase=0;phase<4;++phase)
            {
                __nv_fp8_e4m3 val;
                val.__x=codes[phase];
                const int outputX=x*2+(phase&1);
                const int outputY=y*2+(phase>>1);
                const size_t destination=(
                    size_t(outputY)*(lowWidth*2)+outputX)*Channels+c;
                const float combined=
                    float(val)*expandedScale+
                    __half2float(__ushort_as_half(skipBits[destination]));
                outputBits[destination]=
                    __half_as_ushort(__float2half_rn(combined));
            }
        }

        template <bool Fp8Input>
        __global__ void ComposeTailKernel(
            const float* accumulator, float inputScale,
            const float* weightScale, const float* bias,
            const ushort4* edge,int packedWidth,int outputWidth,
            int outputHeight, float edgePhaseMaxShift,
            cudaSurfaceObject_t output)
        {
            const int x=int(blockIdx.x*blockDim.x+threadIdx.x);
            const int y=int(blockIdx.y*blockDim.y+threadIdx.y);
            if(x>=outputWidth||y>=outputHeight) return;
            constexpr int trim=16;
            const int fullX=x+trim, fullY=y+trim;
            const int packedX=fullX>>2, packedY=fullY>>2;
            const int phase=((fullY&3)<<2)|(fullX&3);
            const size_t packedBase=(size_t(packedY)*packedWidth+packedX)*64;
            float values[4];
#pragma unroll
            for(int group=0;group<4;++group)
            {
                const int channel=group*16+phase;
                float value=accumulator[packedBase+channel];
                if constexpr(Fp8Input) value*=inputScale*weightScale[channel];
                values[group]=value+bias[channel];
            }
            const ushort4 basePacked=surf2Dread<ushort4>(
                output,x*int(sizeof(ushort4)),y);
            const ushort4 edgePacked=edge[size_t(y)*outputWidth+x];
            const float baseR=__half2float(__ushort_as_half(basePacked.x));
            const float baseG=__half2float(__ushort_as_half(basePacked.y));
            const float baseB=__half2float(__ushort_as_half(basePacked.z));
            const float edgeR=__half2float(__ushort_as_half(edgePacked.x));
            const float edgeG=__half2float(__ushort_as_half(edgePacked.y));
            const float edgeB=__half2float(__ushort_as_half(edgePacked.z));
            const float shift=__tanhf(values[3])*edgePhaseMaxShift;
            const float r=fminf(1.0f,fmaxf(0.0f,baseR+values[0]+shift*edgeR));
            const float g=fminf(1.0f,fmaxf(0.0f,baseG+values[1]+shift*edgeG));
            const float b=fminf(1.0f,fmaxf(0.0f,baseB+values[2]+shift*edgeB));
            surf2Dwrite(make_ushort4(
                __half_as_ushort(__float2half_rn(r)),
                __half_as_ushort(__float2half_rn(g)),
                __half_as_ushort(__float2half_rn(b)),
                __half_as_ushort(__float2half_rn(1.0f))),
                output,x*int(sizeof(ushort4)),y);
        }

        __global__ void ComposeTailHalfKernel(
            const Half* tail,const ushort4* edge,int packedWidth,
            int outputWidth,int outputHeight,float edgePhaseMaxShift,
            cudaSurfaceObject_t output)
        {
            const int x=int(blockIdx.x*blockDim.x+threadIdx.x);
            const int y=int(blockIdx.y*blockDim.y+threadIdx.y);
            if(x>=outputWidth||y>=outputHeight) return;
            constexpr int trim=16;
            const int fullX=x+trim,fullY=y+trim;
            const int packedX=fullX>>2,packedY=fullY>>2;
            const int phase=((fullY&3)<<2)|(fullX&3);
            const size_t base=(size_t(packedY)*packedWidth+packedX)*64;
            const float rResidual=float(tail[base+phase]);
            const float gResidual=float(tail[base+16+phase]);
            const float bResidual=float(tail[base+32+phase]);
            const float phaseValue=float(tail[base+48+phase]);
            const ushort4 basePacked=surf2Dread<ushort4>(
                output,x*int(sizeof(ushort4)),y);
            const ushort4 edgePacked=edge[size_t(y)*outputWidth+x];
            const float baseR=__half2float(__ushort_as_half(basePacked.x));
            const float baseG=__half2float(__ushort_as_half(basePacked.y));
            const float baseB=__half2float(__ushort_as_half(basePacked.z));
            const float edgeR=__half2float(__ushort_as_half(edgePacked.x));
            const float edgeG=__half2float(__ushort_as_half(edgePacked.y));
            const float edgeB=__half2float(__ushort_as_half(edgePacked.z));
            const float shift=__tanhf(phaseValue)*edgePhaseMaxShift;
            const float r=fminf(1.0f,fmaxf(0.0f,
                baseR+rResidual+shift*edgeR));
            const float g=fminf(1.0f,fmaxf(0.0f,
                baseG+gResidual+shift*edgeG));
            const float b=fminf(1.0f,fmaxf(0.0f,
                baseB+bResidual+shift*edgeB));
            surf2Dwrite(make_ushort4(
                __half_as_ushort(__float2half_rn(r)),
                __half_as_ushort(__float2half_rn(g)),
                __half_as_ushort(__float2half_rn(b)),
                __half_as_ushort(__float2half_rn(1.0f))),
                output,x*int(sizeof(ushort4)),y);
        }
    }

    // RGBY/PU2 directly from the imported source: one surface read per source
    // pixel, no intermediate RGBY image and no EASU/base/edge allocation.
    __device__ __forceinline__ uint32_t PackHalfPair(float a,float b)
    {
        const __half2_raw value=__floats2half2_rn(a,b);
        return uint32_t(value.x)|(uint32_t(value.y)<<16);
    }

    __device__ __forceinline__ uint32_t PackFourFp8(float a,float b,float c,float d,float reciprocal)
    {
        const __nv_fp8_e4m3 qa(a*reciprocal),qb(b*reciprocal),qc(c*reciprocal),qd(d*reciprocal);
        return uint32_t(qa.__x)|(uint32_t(qb.__x)<<8)|(uint32_t(qc.__x)<<16)|(uint32_t(qd.__x)<<24);
    }

    template <typename T, int StoredChannels = 16, bool Feature8 = false>
    __global__ void PackRgbYKernel(cudaSurfaceObject_t source, int width, int height,
        int packedWidth, int packedHeight, int padX, int padY, T* packed, float reciprocal)
    {
        const int pixel = int(blockIdx.x * blockDim.x + threadIdx.x);
        if (pixel >= packedWidth * packedHeight) return;
        const int x = (pixel % packedWidth) * 2 - padX;
        const int y = (pixel / packedWidth) * 2 - padY;
        float3 rgb[4];
        float luma[4];

        if constexpr (Feature8)
        {
            float lumaGrid[4][4];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    const float3 s = ReadSource(source, width, height, y - 1 + r, x - 1 + c);
                    const float ly = __fadd_rn(__fadd_rn(__fmul_rn(s.x, 0.2126f),
                        __fmul_rn(s.y, 0.7152f)), __fmul_rn(s.z, 0.0722f));
                    lumaGrid[r][c] = ly;
                    if (r == 1 && c == 1) { rgb[0] = s; luma[0] = ly; }
                    else if (r == 1 && c == 2) { rgb[1] = s; luma[1] = ly; }
                    else if (r == 2 && c == 1) { rgb[2] = s; luma[2] = ly; }
                    else if (r == 2 && c == 2) { rgb[3] = s; luma[3] = ly; }
                }
            }

            float dx[4], dy[4], lap[4], mag[4];
            #pragma unroll
            for (int phase = 0; phase < 4; ++phase)
            {
                const int pr = (phase >> 1) + 1;
                const int pc = (phase & 1) + 1;
                const float ul = lumaGrid[pr - 1][pc - 1];
                const float uc = lumaGrid[pr - 1][pc];
                const float ur = lumaGrid[pr - 1][pc + 1];
                const float cl = lumaGrid[pr][pc - 1];
                const float cc = lumaGrid[pr][pc];
                const float cr = lumaGrid[pr][pc + 1];
                const float dl = lumaGrid[pr + 1][pc - 1];
                const float dc = lumaGrid[pr + 1][pc];
                const float dr = lumaGrid[pr + 1][pc + 1];

                const float sdx = (-ul + ur - 2.0f * cl + 2.0f * cr - dl + dr) * 0.125f;
                const float sdy = (-ul - 2.0f * uc - ur + dl + 2.0f * dc + dr) * 0.125f;
                const float slap = (4.0f * cc - cl - cr - uc - dc) * 0.25f;
                dx[phase] = sdx;
                dy[phase] = sdy;
                lap[phase] = slap;
                mag[phase] = fabsf(sdx) + fabsf(sdy);
            }

            if constexpr (std::is_same_v<T, Half>)
            {
                uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 4;
                destination[0] = make_uint4(PackHalfPair(rgb[0].x, rgb[1].x), PackHalfPair(rgb[2].x, rgb[3].x),
                    PackHalfPair(rgb[0].y, rgb[1].y), PackHalfPair(rgb[2].y, rgb[3].y));
                destination[1] = make_uint4(PackHalfPair(rgb[0].z, rgb[1].z), PackHalfPair(rgb[2].z, rgb[3].z),
                    PackHalfPair(luma[0], luma[1]), PackHalfPair(luma[2], luma[3]));
                destination[2] = make_uint4(PackHalfPair(dx[0], dx[1]), PackHalfPair(dx[2], dx[3]),
                    PackHalfPair(dy[0], dy[1]), PackHalfPair(dy[2], dy[3]));
                destination[3] = make_uint4(PackHalfPair(lap[0], lap[1]), PackHalfPair(lap[2], lap[3]),
                    PackHalfPair(mag[0], mag[1]), PackHalfPair(mag[2], mag[3]));
            }
            else
            {
                uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 2;
                destination[0] = make_uint4(
                    PackFourFp8(rgb[0].x, rgb[1].x, rgb[2].x, rgb[3].x, reciprocal),
                    PackFourFp8(rgb[0].y, rgb[1].y, rgb[2].y, rgb[3].y, reciprocal),
                    PackFourFp8(rgb[0].z, rgb[1].z, rgb[2].z, rgb[3].z, reciprocal),
                    PackFourFp8(luma[0], luma[1], luma[2], luma[3], reciprocal));
                destination[1] = make_uint4(
                    PackFourFp8(dx[0], dx[1], dx[2], dx[3], reciprocal),
                    PackFourFp8(dy[0], dy[1], dy[2], dy[3], reciprocal),
                    PackFourFp8(lap[0], lap[1], lap[2], lap[3], reciprocal),
                    PackFourFp8(mag[0], mag[1], mag[2], mag[3], reciprocal));
            }
        }
        else
        {
            #pragma unroll
            for (int phase = 0; phase < 4; ++phase)
            {
                rgb[phase] = ReadSource(source, width, height, y + (phase >> 1), x + (phase & 1));
                // Explicit rounding follows the separate Torch RGBY arithmetic.
                luma[phase] = __fadd_rn(__fadd_rn(__fmul_rn(rgb[phase].x, 0.2126f),
                    __fmul_rn(rgb[phase].y, 0.7152f)), __fmul_rn(rgb[phase].z, 0.0722f));
            }
            // Aligned vector stores replace sixteen strided scalar stores.
            // NHWC channel order remains R00,R01,R10,R11,G00,...,Y11.
            if constexpr (std::is_same_v<T, Half>)
            {
                uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 2;
                destination[0] = make_uint4(PackHalfPair(rgb[0].x, rgb[1].x), PackHalfPair(rgb[2].x, rgb[3].x),
                    PackHalfPair(rgb[0].y, rgb[1].y), PackHalfPair(rgb[2].y, rgb[3].y));
                destination[1] = make_uint4(PackHalfPair(rgb[0].z, rgb[1].z), PackHalfPair(rgb[2].z, rgb[3].z),
                    PackHalfPair(luma[0], luma[1]), PackHalfPair(luma[2], luma[3]));
            }
            else
            {
                if constexpr (StoredChannels == 32)
                {
                    uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 2;
                    destination[0] = make_uint4(
                        PackFourFp8(rgb[0].x, rgb[1].x, rgb[2].x, rgb[3].x, reciprocal),
                        PackFourFp8(rgb[0].y, rgb[1].y, rgb[2].y, rgb[3].y, reciprocal),
                        PackFourFp8(rgb[0].z, rgb[1].z, rgb[2].z, rgb[3].z, reciprocal),
                        PackFourFp8(luma[0], luma[1], luma[2], luma[3], reciprocal));
                    destination[1] = make_uint4(0, 0, 0, 0);
                }
                else
                {
                    reinterpret_cast<uint4*>(packed)[pixel] = make_uint4(
                        PackFourFp8(rgb[0].x, rgb[1].x, rgb[2].x, rgb[3].x, reciprocal),
                        PackFourFp8(rgb[0].y, rgb[1].y, rgb[2].y, rgb[3].y, reciprocal),
                        PackFourFp8(rgb[0].z, rgb[1].z, rgb[2].z, rgb[3].z, reciprocal),
                        PackFourFp8(luma[0], luma[1], luma[2], luma[3], reciprocal));
                }
            }
        }
    }

    template <typename T>
    __global__ void PackLumaKernel(cudaSurfaceObject_t source, int width, int height,
        int packedWidth, int packedHeight, int padX, int padY, T* packed, float reciprocal)
    {
        const int pixel = int(blockIdx.x * blockDim.x + threadIdx.x);
        if (pixel >= packedWidth * packedHeight) return;
        const int x = (pixel % packedWidth) * 2 - padX;
        const int y = (pixel / packedWidth) * 2 - padY;
        float luma[4];
        #pragma unroll
        for (int phase = 0; phase < 4; ++phase)
        {
            const float3 rgb = ReadSource(source, width, height, y + (phase >> 1), x + (phase & 1));
            luma[phase] = __fadd_rn(__fadd_rn(__fmul_rn(rgb.x, 0.2126f),
                __fmul_rn(rgb.y, 0.7152f)), __fmul_rn(rgb.z, 0.0722f));
        }
        if constexpr (std::is_same_v<T, Half>)
        {
            uint2* destination = reinterpret_cast<uint2*>(packed) + pixel;
            *destination = make_uint2(PackHalfPair(luma[0], luma[1]), PackHalfPair(luma[2], luma[3]));
        }
        else
        {
            uint32_t* destination = reinterpret_cast<uint32_t*>(packed) + pixel;
            *destination = PackFourFp8(luma[0], luma[1], luma[2], luma[3], reciprocal);
        }
    }

    static void PackRefineWeights(const half* w_oihw, half* out_packed)
    {
        for (int r = 0; r < 3; ++r)
        {
            for (int s = 0; s < 3; ++s)
            {
                int tap = r * 3 + s;
                for (int chGrp = 0; chGrp < 4; ++chGrp)
                {
                    for (int kChunk = 0; kChunk < 2; ++kChunk)
                    {
                        for (int coSub = 0; coSub < 16; ++coSub)
                        {
                            for (int ciSub = 0; ciSub < 16; ++ciSub)
                            {
                                int co = chGrp * 16 + coSub;
                                int ci = kChunk * 16 + ciSub;
                                int oihwIdx = ((co * 32 + ci) * 3 + r) * 3 + s;
                                int packedIdx = (((tap * 4 + chGrp) * 2 + kChunk) * 16 + coSub) * 16 + ciSub;
                                out_packed[packedIdx] = w_oihw[oihwIdx];
                            }
                        }
                    }
                }
            }
        }
    }

    __global__ void __launch_bounds__(256, 2) RefineFastWmmaKernel(
        const half* __restrict__ input,
        const half* __restrict__ packedWeights,
        const float* __restrict__ bias,
        half* __restrict__ output,
        int width, int height)
    {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        constexpr int PATCH_W = TILE_W + 2; // 18
        constexpr int PATCH_H = TILE_H + 2; // 10
        constexpr int C_IN = 32;
        constexpr int C_OUT = 64;

        int tileX = blockIdx.x * TILE_W;
        int tileY = blockIdx.y * TILE_H;

        struct SharedMem {
            half patch[PATCH_H][PATCH_W][C_IN]; // 10 * 18 * 32 * 2 = 11,520 bytes
            float temp[8][16][16];          // 8 * 16 * 16 * 4 = 8,192 bytes
            half out_half[8][16][64];       // 8 * 16 * 64 * 2 = 16,384 bytes
        };
        __shared__ alignas(16) SharedMem smem;

        const int tid = threadIdx.y * 32 + threadIdx.x;
        const int totalThreads = 256;

        constexpr int TOTAL_U4 = (PATCH_H * PATCH_W * C_IN) / 8; // 720
        for (int idx = tid; idx < TOTAL_U4; idx += totalThreads)
        {
            int pPix = idx / 4;
            int cChunk = (idx % 4) * 8;
            int py = pPix / PATCH_W;
            int px = pPix % PATCH_W;
            int gy = tileY + py - 1;
            int gx = tileX + px - 1;

            if (gy >= 0 && gy < height && gx >= 0 && gx < width)
            {
                *reinterpret_cast<uint4*>(&smem.patch[py][px][cChunk]) = 
                    *reinterpret_cast<const uint4*>(input + (gy * width + gx) * C_IN + cChunk);
            }
            else
            {
                *reinterpret_cast<uint4*>(&smem.patch[py][px][cChunk]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        const int warpId = threadIdx.y;
        const int laneId = threadIdx.x;
        int row = warpId;
        int gy = tileY + row;

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[4];
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            nvcuda::wmma::fill_fragment(acc[g], 0.0f);
        }

        if (gy < height)
        {
            #pragma unroll
            for (int r = 0; r < 3; ++r)
            {
                int py = row + r;
                #pragma unroll
                for (int s = 0; s < 3; ++s)
                {
                    int tap = r * 3 + s;
                    #pragma unroll
                    for (int kChunk = 0; kChunk < 2; ++kChunk)
                    {
                        int cinBase = kChunk * 16;
                        const half* aPtr = &smem.patch[py][s][cinBase];
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
                        nvcuda::wmma::load_matrix_sync(a_frag, aPtr, C_IN);

                        #pragma unroll
                        for (int chGrp = 0; chGrp < 4; ++chGrp)
                        {
                            const half* bPtr = packedWeights + (((tap * 4 + chGrp) * 2 + kChunk) * 16) * 16;
                            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
                            nvcuda::wmma::load_matrix_sync(b_frag, bPtr, 16);

                            nvcuda::wmma::mma_sync(acc[chGrp], a_frag, b_frag, acc[chGrp]);
                        }
                    }
                }
            }

            #pragma unroll
            for (int chGrp = 0; chGrp < 4; ++chGrp)
            {
                nvcuda::wmma::store_matrix_sync(&smem.temp[warpId][0][0], acc[chGrp], 16, nvcuda::wmma::mem_row_major);
                __syncwarp();

                #pragma unroll
                for (int item = 0; item < 8; ++item)
                {
                    int elemIdx = laneId * 8 + item;
                    int p = elemIdx / 16;
                    int cSub = elemIdx % 16;
                    int co = chGrp * 16 + cSub;

                    float v = smem.temp[warpId][p][cSub] + bias[co];
                    float act = (v > 0.0f) ? v : (v * 0.1f);
                    smem.out_half[warpId][p][co] = __float2half_rn(act);
                }
                __syncwarp();
            }

            #pragma unroll
            for (int step = 0; step < 4; ++step)
            {
                int p = step * 4 + (laneId / 8);
                int c = (laneId % 8) * 8;
                int gx = tileX + p;
                if (gx < width)
                {
                    *reinterpret_cast<uint4*>(output + (gy * width + gx) * C_OUT + c) = 
                        *reinterpret_cast<const uint4*>(&smem.out_half[warpId][p][c]);
                }
            }
        }
    }

    __global__ void __launch_bounds__(256, 2) RefineWmmaPairMoEKernel(
        const Half* __restrict__ input,
        const Half* __restrict__ weights,
        const float* __restrict__ bias,
        const float* __restrict__ routeThreshold,
        Half* __restrict__ decOut,
        Half* __restrict__ routeOut,
        int width, int height)
    {
        constexpr int TILE_W = 16;
        constexpr int TILE_H = 8;
        constexpr int PATCH_W = TILE_W + 2;
        constexpr int PATCH_H = TILE_H + 2;
        constexpr int C_IN = 32;
        constexpr int C_OUT = 64;

        int tileX = blockIdx.x * TILE_W;
        int tileY = blockIdx.y * TILE_H;

        __shared__ alignas(16) half s_patch[PATCH_H][PATCH_W][C_IN];
        __shared__ alignas(16) half s_warp[8][16][16];

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int totalThreads = blockDim.x * blockDim.y;

        constexpr int CHUNKS_8 = C_IN / 8;
        for (int idx = tid; idx < PATCH_H * PATCH_W * CHUNKS_8; idx += totalThreads)
        {
            int pPix = idx / CHUNKS_8;
            int cChunk = (idx % CHUNKS_8) * 8;
            int py = pPix / PATCH_W;
            int px = pPix % PATCH_W;
            int gy = tileY + py - 1;
            int gx = tileX + px - 1;

            if (gy >= 0 && gy < height && gx >= 0 && gx < width)
            {
                *reinterpret_cast<uint4*>(&s_patch[py][px][cChunk]) = 
                    *reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(input) + (gy * width + gx) * C_IN + cChunk);
            }
            else
            {
                *reinterpret_cast<uint4*>(&s_patch[py][px][cChunk]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        const int warpId = tid / 32;
        const int laneId = tid & 31;
        const int numWarps = totalThreads / 32;
        #pragma unroll 1
        for (int unit = warpId; unit < 32; unit += numWarps)
        {
            int row = unit / 4;
            int chGroup = unit % 4;
            int coutBase = chGroup * 16;

            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> acc;
            nvcuda::wmma::fill_fragment(acc, __float2half(0.0f));

            #pragma unroll
            for (int r = 0; r < 3; ++r)
            {
                #pragma unroll
                for (int s = 0; s < 3; ++s)
                {
                    int tap = r * 3 + s;
                    #pragma unroll
                    for (int kChunk = 0; kChunk < 2; ++kChunk)
                    {
                        int cinBase = kChunk * 16;
                        const half* aPtr = &s_patch[row + r][s][cinBase];
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
                        nvcuda::wmma::load_matrix_sync(a_frag, aPtr, C_IN);

                        const half* bPtr = reinterpret_cast<const half*>(weights) + (coutBase * 9 + tap) * 32 + cinBase;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
                        nvcuda::wmma::load_matrix_sync(b_frag, bPtr, 9 * 32);

                        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
                    }
                }
            }

            nvcuda::wmma::store_matrix_sync(&s_warp[warpId][0][0], acc, 16, nvcuda::wmma::mem_row_major);
            __syncwarp();

            int gy = tileY + row;
            if (gy < height)
            {
                int px = laneId / 2;
                int sub = (laneId & 1) * 8;
                int c = coutBase + sub;
                int gx = tileX + px;

                if (gx < width)
                {
                    uint4 val4 = *reinterpret_cast<const uint4*>(&s_warp[warpId][px][sub]);
                    half* hVals = reinterpret_cast<half*>(&val4);
                    half dVals[8];
                    half rVals[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                    {
                        float v = __half2float(hVals[i]) + bias[c + i];
                        float lrelu = (v > 0.0f) ? v : (v * 0.1f);
                        dVals[i] = __float2half_rn(lrelu);
                        rVals[i] = __float2half_rn(v - routeThreshold[c + i]);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<half*>(decOut) + (gy * width + gx) * C_OUT + c) = *reinterpret_cast<uint4*>(dVals);
                    *reinterpret_cast<uint4*>(reinterpret_cast<half*>(routeOut) + (gy * width + gx) * C_OUT + c) = *reinterpret_cast<uint4*>(rVals);
                }
            }
            __syncwarp();
        }
    }

    __global__ void RefineDualOutputEpilogueKernel(
        const Half* __restrict__ routeSource,
        const float* __restrict__ routeThreshold,
        Half* __restrict__ decOut,
        Half* __restrict__ routeMetaOut,
        int pixelCount)
    {
        const int pixel = int(blockIdx.x * blockDim.x + threadIdx.x);
        if (pixel >= pixelCount) return;
        const size_t base = size_t(pixel) * 64;
        const auto* srcHalf = reinterpret_cast<const __half*>(routeSource + base);
        auto* decHalf = reinterpret_cast<__half*>(decOut + base);
        auto* routeHalf = reinterpret_cast<__half*>(routeMetaOut + base);
        #pragma unroll
        for (int group = 0; group < 4; ++group)
        {
            const uint4 packed = reinterpret_cast<const uint4*>(srcHalf)[group];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const unsigned bits = j == 0 ? packed.x : (j == 1 ? packed.y : (j == 2 ? packed.z : packed.w));
                const int ch0 = group * 16 + j * 2;
                const int ch1 = ch0 + 1;
                const float s0 = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
                const float s1 = __half2float(__ushort_as_half(static_cast<unsigned short>(bits >> 16)));
                const float d0 = s0 > 0.0f ? s0 : s0 * 0.1f;
                const float d1 = s1 > 0.0f ? s1 : s1 * 0.1f;
                const float r0 = s0 - routeThreshold[ch0];
                const float r1 = s1 - routeThreshold[ch1];
                decHalf[ch0] = __float2half_rn(d0);
                decHalf[ch1] = __float2half_rn(d1);
                routeHalf[ch0] = __float2half_rn(r0);
                routeHalf[ch1] = __float2half_rn(r1);
            }
        }
    }

    __device__ __forceinline__ float ComputeAnalysisLuma(float3 s, const LearnedAnalysisStencil& stencil)
    {
        return __fadd_rn(__fadd_rn(__fmul_rn(s.x, stencil.rgbMix[0]), __fmul_rn(s.y, stencil.rgbMix[1])), __fmul_rn(s.z, stencil.rgbMix[2]));
    }

    template <typename T>
    __global__ void PackRgbYFeature8TiledKernel(cudaSurfaceObject_t source, int width, int height,
        int packedWidth, int packedHeight, int padX, int padY, T* packed, float reciprocal,
        LearnedAnalysisStencil stencil = {})
    {
        constexpr int BW = 16, BH = 16;
        __shared__ float tileLuma[34][36]; // 34 rows x 36 cols (padded to avoid bank conflicts)

        const int tid = int(threadIdx.y * BW + threadIdx.x);
        const int bx = int(blockIdx.x * BW);
        const int by = int(blockIdx.y * BH);

        const int tx = int(threadIdx.x);
        const int ty = int(threadIdx.y);
        const int px = bx + tx;
        const int py = by + ty;
        const bool activePixel = (px < packedWidth && py < packedHeight);

        // 1. Each thread loads its own 2x2 source quad (interior of block)
        const int sx = px * 2 - padX;
        const int sy = py * 2 - padY;
        const float3 s0 = ReadSource(source, width, height, sy, sx);
        const float3 s1 = ReadSource(source, width, height, sy, sx + 1);
        const float3 s2 = ReadSource(source, width, height, sy + 1, sx);
        const float3 s3 = ReadSource(source, width, height, sy + 1, sx + 1);

        const float ly0 = ComputeAnalysisLuma(s0, stencil);
        const float ly1 = ComputeAnalysisLuma(s1, stencil);
        const float ly2 = ComputeAnalysisLuma(s2, stencil);
        const float ly3 = ComputeAnalysisLuma(s3, stencil);

        tileLuma[ty * 2 + 1][tx * 2 + 1] = ly0;
        tileLuma[ty * 2 + 1][tx * 2 + 2] = ly1;
        tileLuma[ty * 2 + 2][tx * 2 + 1] = ly2;
        tileLuma[ty * 2 + 2][tx * 2 + 2] = ly3;

        // 2. Load the 1-pixel halo around the 32x32 block (132 halo elements loaded by threads 0..131)
        if (tid < 34)
        {
            // Top row: sy = by * 2 - padY - 1, sx = bx * 2 - padX - 1 + tid
            const float3 h = ReadSource(source, width, height, by * 2 - padY - 1, bx * 2 - padX - 1 + tid);
            tileLuma[0][tid] = ComputeAnalysisLuma(h, stencil);
        }
        else if (tid < 68)
        {
            // Bottom row: sy = by * 2 - padY + 32, sx = bx * 2 - padX - 1 + (tid - 34)
            const int c = tid - 34;
            const float3 h = ReadSource(source, width, height, by * 2 - padY + 32, bx * 2 - padX - 1 + c);
            tileLuma[33][c] = ComputeAnalysisLuma(h, stencil);
        }
        else if (tid < 100)
        {
            // Left col: sy = by * 2 - padY + (tid - 68), sx = bx * 2 - padX - 1
            const int r = tid - 68 + 1; // row in 1..32
            const float3 h = ReadSource(source, width, height, by * 2 - padY + (tid - 68), bx * 2 - padX - 1);
            tileLuma[r][0] = ComputeAnalysisLuma(h, stencil);
        }
        else if (tid < 132)
        {
            // Right col: sy = by * 2 - padY + (tid - 100), sx = bx * 2 - padX + 32
            const int r = tid - 100 + 1; // row in 1..32
            const float3 h = ReadSource(source, width, height, by * 2 - padY + (tid - 100), bx * 2 - padX + 32);
            tileLuma[r][33] = ComputeAnalysisLuma(h, stencil);
        }

        __syncthreads();

        if (!activePixel) return;

        // 3. Load 4x4 luma grid from shared memory
        const int r_base = ty * 2 + 1;
        const int c_base = tx * 2 + 1;
        float lumaGrid[4][4];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                lumaGrid[r][c] = tileLuma[r_base - 1 + r][c_base - 1 + c];
            }
        }

        float dx[4], dy[4], lap[4], mag[4];
        #pragma unroll
        for (int phase = 0; phase < 4; ++phase)
        {
            const int pr = (phase >> 1) + 1;
            const int pc = (phase & 1) + 1;
            const float ul = lumaGrid[pr - 1][pc - 1];
            const float uc = lumaGrid[pr - 1][pc];
            const float ur = lumaGrid[pr - 1][pc + 1];
            const float cl = lumaGrid[pr][pc - 1];
            const float cc = lumaGrid[pr][pc];
            const float cr = lumaGrid[pr][pc + 1];
            const float dl = lumaGrid[pr + 1][pc - 1];
            const float dc = lumaGrid[pr + 1][pc];
            const float dr = lumaGrid[pr + 1][pc + 1];

            const float sdx = (ur - ul) * stencil.dxRows[0] + (cr - cl) * stencil.dxRows[1] + (dr - dl) * stencil.dxRows[2];
            const float sdy = (dl - ul) * stencil.dyCols[0] + (dc - uc) * stencil.dyCols[1] + (dr - ur) * stencil.dyCols[2];
            const float ring = cl + cr + uc + dc;
            const float slap = cc * stencil.lap[0] + ring * stencil.lap[1];
            dx[phase] = sdx;
            dy[phase] = sdy;
            lap[phase] = slap;
            mag[phase] = fabsf(sdx) + fabsf(sdy);
        }

        const int pixel = py * packedWidth + px;

        if constexpr (std::is_same_v<T, Half>)
        {
            uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 4;
            destination[0] = make_uint4(PackHalfPair(s0.x, s1.x), PackHalfPair(s2.x, s3.x),
                PackHalfPair(s0.y, s1.y), PackHalfPair(s2.y, s3.y));
            destination[1] = make_uint4(PackHalfPair(s0.z, s1.z), PackHalfPair(s2.z, s3.z),
                PackHalfPair(ly0, ly1), PackHalfPair(ly2, ly3));
            destination[2] = make_uint4(PackHalfPair(dx[0], dx[1]), PackHalfPair(dx[2], dx[3]),
                PackHalfPair(dy[0], dy[1]), PackHalfPair(dy[2], dy[3]));
            destination[3] = make_uint4(PackHalfPair(lap[0], lap[1]), PackHalfPair(lap[2], lap[3]),
                PackHalfPair(mag[0], mag[1]), PackHalfPair(mag[2], mag[3]));
        }
        else
        {
            uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 2;
            destination[0] = make_uint4(
                PackFourFp8(s0.x, s1.x, s2.x, s3.x, reciprocal),
                PackFourFp8(s0.y, s1.y, s2.y, s3.y, reciprocal),
                PackFourFp8(s0.z, s1.z, s2.z, s3.z, reciprocal),
                PackFourFp8(ly0, ly1, ly2, ly3, reciprocal));
            destination[1] = make_uint4(
                PackFourFp8(dx[0], dx[1], dx[2], dx[3], reciprocal),
                PackFourFp8(dy[0], dy[1], dy[2], dy[3], reciprocal),
                PackFourFp8(lap[0], lap[1], lap[2], lap[3], reciprocal),
                PackFourFp8(mag[0], mag[1], mag[2], mag[3], reciprocal));
        }
    }

    __device__ __forceinline__ float ComputeAnalysis4DirLuma(float3 s, const LearnedAnalysis4Dir& stencil)
    {
        return __fadd_rn(__fadd_rn(__fmul_rn(s.x, stencil.rgbMix[0]), __fmul_rn(s.y, stencil.rgbMix[1])), __fmul_rn(s.z, stencil.rgbMix[2]));
    }

    __device__ __forceinline__ float3 ComputeLearnedRgb(float3 s, const LearnedAnalysis4Dir& stencil)
    {
        float r = __fadd_rn(__fadd_rn(__fmul_rn(s.x, stencil.rgbBasis[0]), __fmul_rn(s.y, stencil.rgbBasis[1])), __fmul_rn(s.z, stencil.rgbBasis[2]));
        float g = __fadd_rn(__fadd_rn(__fmul_rn(s.x, stencil.rgbBasis[3]), __fmul_rn(s.y, stencil.rgbBasis[4])), __fmul_rn(s.z, stencil.rgbBasis[5]));
        float b = __fadd_rn(__fadd_rn(__fmul_rn(s.x, stencil.rgbBasis[6]), __fmul_rn(s.y, stencil.rgbBasis[7])), __fmul_rn(s.z, stencil.rgbBasis[8]));
        return make_float3(r, g, b);
    }

    template <typename T>
    __global__ void PackRgbYLearnedRgb4DirTiledKernel(cudaSurfaceObject_t source, int width, int height,
        int packedWidth, int packedHeight, int padX, int padY, T* packed, float reciprocal,
        LearnedAnalysis4Dir stencil = {})
    {
        constexpr int BW = 16, BH = 16;
        __shared__ float tileLuma[34][36];

        const int tid = int(threadIdx.y * BW + threadIdx.x);
        const int bx = int(blockIdx.x * BW);
        const int by = int(blockIdx.y * BH);

        const int tx = int(threadIdx.x);
        const int ty = int(threadIdx.y);
        const int px = bx + tx;
        const int py = by + ty;
        const bool activePixel = (px < packedWidth && py < packedHeight);

        // 1. Each thread loads its own 2x2 source quad
        const int sx = px * 2 - padX;
        const int sy = py * 2 - padY;
        const float3 s0 = ReadSource(source, width, height, sy, sx);
        const float3 s1 = ReadSource(source, width, height, sy, sx + 1);
        const float3 s2 = ReadSource(source, width, height, sy + 1, sx);
        const float3 s3 = ReadSource(source, width, height, sy + 1, sx + 1);

        const float ly0 = ComputeAnalysis4DirLuma(s0, stencil);
        const float ly1 = ComputeAnalysis4DirLuma(s1, stencil);
        const float ly2 = ComputeAnalysis4DirLuma(s2, stencil);
        const float ly3 = ComputeAnalysis4DirLuma(s3, stencil);

        tileLuma[ty * 2 + 1][tx * 2 + 1] = ly0;
        tileLuma[ty * 2 + 1][tx * 2 + 2] = ly1;
        tileLuma[ty * 2 + 2][tx * 2 + 1] = ly2;
        tileLuma[ty * 2 + 2][tx * 2 + 2] = ly3;

        // 2. Load the 1-pixel halo around the 32x32 block (132 halo elements)
        if (tid < 34)
        {
            const float3 h = ReadSource(source, width, height, by * 2 - padY - 1, bx * 2 - padX - 1 + tid);
            tileLuma[0][tid] = ComputeAnalysis4DirLuma(h, stencil);
        }
        else if (tid < 68)
        {
            const int c = tid - 34;
            const float3 h = ReadSource(source, width, height, by * 2 - padY + 32, bx * 2 - padX - 1 + c);
            tileLuma[33][c] = ComputeAnalysis4DirLuma(h, stencil);
        }
        else if (tid < 100)
        {
            const int r = tid - 68 + 1;
            const float3 h = ReadSource(source, width, height, by * 2 - padY + (tid - 68), bx * 2 - padX - 1);
            tileLuma[r][0] = ComputeAnalysis4DirLuma(h, stencil);
        }
        else if (tid < 132)
        {
            const int r = tid - 100 + 1;
            const float3 h = ReadSource(source, width, height, by * 2 - padY + (tid - 100), bx * 2 - padX + 32);
            tileLuma[r][33] = ComputeAnalysis4DirLuma(h, stencil);
        }

        __syncthreads();

        if (!activePixel) return;

        // Compute learned RGB for the 4 source pixels
        const float3 lr0 = ComputeLearnedRgb(s0, stencil);
        const float3 lr1 = ComputeLearnedRgb(s1, stencil);
        const float3 lr2 = ComputeLearnedRgb(s2, stencil);
        const float3 lr3 = ComputeLearnedRgb(s3, stencil);

        // 3. Load 4x4 luma grid from shared memory
        const int r_base = ty * 2 + 1;
        const int c_base = tx * 2 + 1;
        float lumaGrid[4][4];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                lumaGrid[r][c] = tileLuma[r_base - 1 + r][c_base - 1 + c];
            }
        }

        float dx[4], dy[4], diag45[4], diag135[4];
        #pragma unroll
        for (int phase = 0; phase < 4; ++phase)
        {
            const int pr = (phase >> 1) + 1;
            const int pc = (phase & 1) + 1;
            const float ul = lumaGrid[pr - 1][pc - 1];
            const float uc = lumaGrid[pr - 1][pc];
            const float ur = lumaGrid[pr - 1][pc + 1];
            const float cl = lumaGrid[pr][pc - 1];
            const float cr = lumaGrid[pr][pc + 1];
            const float dl = lumaGrid[pr + 1][pc - 1];
            const float dc = lumaGrid[pr + 1][pc];
            const float dr = lumaGrid[pr + 1][pc + 1];

            const float hmid = cr - cl;
            const float vmid = dc - uc;

            dx[phase] = (ur - ul) * stencil.dxRows[0] + hmid * stencil.dxRows[1] + (dr - dl) * stencil.dxRows[2];
            dy[phase] = (dl - ul) * stencil.dyCols[0] + vmid * stencil.dyCols[1] + (dr - ur) * stencil.dyCols[2];
            diag45[phase] = (dr - ul) * stencil.diag45[0] + vmid * stencil.diag45[1] + hmid * stencil.diag45[2];
            diag135[phase] = (ur - dl) * stencil.diag135[0] - vmid * stencil.diag135[1] + hmid * stencil.diag135[2];
        }

        const int pixel = py * packedWidth + px;

        if constexpr (std::is_same_v<T, Half>)
        {
            uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 4;
            destination[0] = make_uint4(PackHalfPair(lr0.x, lr1.x), PackHalfPair(lr2.x, lr3.x),
                PackHalfPair(lr0.y, lr1.y), PackHalfPair(lr2.y, lr3.y));
            destination[1] = make_uint4(PackHalfPair(lr0.z, lr1.z), PackHalfPair(lr2.z, lr3.z),
                PackHalfPair(ly0, ly1), PackHalfPair(ly2, ly3));
            destination[2] = make_uint4(PackHalfPair(dx[0], dx[1]), PackHalfPair(dx[2], dx[3]),
                PackHalfPair(dy[0], dy[1]), PackHalfPair(dy[2], dy[3]));
            destination[3] = make_uint4(PackHalfPair(diag45[0], diag45[1]), PackHalfPair(diag45[2], diag45[3]),
                PackHalfPair(diag135[0], diag135[1]), PackHalfPair(diag135[2], diag135[3]));
        }
        else
        {
            uint4* destination = reinterpret_cast<uint4*>(packed) + size_t(pixel) * 2;
            destination[0] = make_uint4(
                PackFourFp8(lr0.x, lr1.x, lr2.x, lr3.x, reciprocal),
                PackFourFp8(lr0.y, lr1.y, lr2.y, lr3.y, reciprocal),
                PackFourFp8(lr0.z, lr1.z, lr2.z, lr3.z, reciprocal),
                PackFourFp8(ly0, ly1, ly2, ly3, reciprocal));
            destination[1] = make_uint4(
                PackFourFp8(dx[0], dx[1], dx[2], dx[3], reciprocal),
                PackFourFp8(dy[0], dy[1], dy[2], dy[3], reciprocal),
                PackFourFp8(diag45[0], diag45[1], diag45[2], diag45[3], reciprocal),
                PackFourFp8(diag135[0], diag135[1], diag135[2], diag135[3], reciprocal));
        }
    }

    __global__ void FusedPackRgbYLearnedRgb4DirAndProjectKernel(
        cudaSurfaceObject_t source, int width, int height,
        int packedWidth, int packedHeight, int padX, int padY,
        half* __restrict__ skipOutput,
        const half* __restrict__ projWeights,
        const float* __restrict__ projBiases,
        const float* __restrict__ projPrelus,
        half* __restrict__ packedOutput = nullptr,
        LearnedAnalysis4Dir stencil = {})
    {
        constexpr int BW = 16, BH = 16;
        __shared__ float tileLuma[34][36];
        __shared__ float s_out[8][32][32]; // 32 KB
        __shared__ float s_projBiases[32];
        __shared__ float s_projPrelus[32];

        const int tid = int(threadIdx.y * BW + threadIdx.x);
        const int bx = int(blockIdx.x * BW);
        const int by = int(blockIdx.y * BH);

        if (tid < 32)
        {
            s_projBiases[tid] = projBiases[tid];
            s_projPrelus[tid] = projPrelus[tid];
        }

        const int tx = int(threadIdx.x);
        const int ty = int(threadIdx.y);
        const int px = bx + tx;
        const int py = by + ty;
        const bool activePixel = (px < packedWidth && py < packedHeight);

        const int sx = px * 2 - padX;
        const int sy = py * 2 - padY;
        const float3 s0 = ReadSource(source, width, height, sy, sx);
        const float3 s1 = ReadSource(source, width, height, sy, sx + 1);
        const float3 s2 = ReadSource(source, width, height, sy + 1, sx);
        const float3 s3 = ReadSource(source, width, height, sy + 1, sx + 1);

        const float ly0 = ComputeAnalysis4DirLuma(s0, stencil);
        const float ly1 = ComputeAnalysis4DirLuma(s1, stencil);
        const float ly2 = ComputeAnalysis4DirLuma(s2, stencil);
        const float ly3 = ComputeAnalysis4DirLuma(s3, stencil);

        tileLuma[ty * 2 + 1][tx * 2 + 1] = ly0;
        tileLuma[ty * 2 + 1][tx * 2 + 2] = ly1;
        tileLuma[ty * 2 + 2][tx * 2 + 1] = ly2;
        tileLuma[ty * 2 + 2][tx * 2 + 2] = ly3;

        if (tid < 34)
        {
            const float3 h = ReadSource(source, width, height, by * 2 - padY - 1, bx * 2 - padX - 1 + tid);
            tileLuma[0][tid] = ComputeAnalysis4DirLuma(h, stencil);
        }
        else if (tid < 68)
        {
            const int c = tid - 34;
            const float3 h = ReadSource(source, width, height, by * 2 - padY + 32, bx * 2 - padX - 1 + c);
            tileLuma[33][c] = ComputeAnalysis4DirLuma(h, stencil);
        }
        else if (tid < 100)
        {
            const int r = tid - 68 + 1;
            const float3 h = ReadSource(source, width, height, by * 2 - padY + (tid - 68), bx * 2 - padX - 1);
            tileLuma[r][0] = ComputeAnalysis4DirLuma(h, stencil);
        }
        else if (tid < 132)
        {
            const int r = tid - 100 + 1;
            const float3 h = ReadSource(source, width, height, by * 2 - padY + (tid - 100), bx * 2 - padX + 32);
            tileLuma[r][33] = ComputeAnalysis4DirLuma(h, stencil);
        }

        __syncthreads();

        half2 in2[16];
        if (activePixel)
        {
            const float3 lr0 = ComputeLearnedRgb(s0, stencil);
            const float3 lr1 = ComputeLearnedRgb(s1, stencil);
            const float3 lr2 = ComputeLearnedRgb(s2, stencil);
            const float3 lr3 = ComputeLearnedRgb(s3, stencil);

            const int r_base = ty * 2 + 1;
            const int c_base = tx * 2 + 1;
            float lumaGrid[4][4];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    lumaGrid[r][c] = tileLuma[r_base - 1 + r][c_base - 1 + c];
                }
            }

            float dx[4], dy[4], diag45[4], diag135[4];
            #pragma unroll
            for (int phase = 0; phase < 4; ++phase)
            {
                const int pr = (phase >> 1) + 1;
                const int pc = (phase & 1) + 1;
                const float ul = lumaGrid[pr - 1][pc - 1];
                const float uc = lumaGrid[pr - 1][pc];
                const float ur = lumaGrid[pr - 1][pc + 1];
                const float cl = lumaGrid[pr][pc - 1];
                const float cr = lumaGrid[pr][pc + 1];
                const float dl = lumaGrid[pr + 1][pc - 1];
                const float dc = lumaGrid[pr + 1][pc];
                const float dr = lumaGrid[pr + 1][pc + 1];

                const float hmid = cr - cl;
                const float vmid = dc - uc;

                dx[phase] = (ur - ul) * stencil.dxRows[0] + hmid * stencil.dxRows[1] + (dr - dl) * stencil.dxRows[2];
                dy[phase] = (dl - ul) * stencil.dyCols[0] + vmid * stencil.dyCols[1] + (dr - ur) * stencil.dyCols[2];
                diag45[phase] = (dr - ul) * stencil.diag45[0] + vmid * stencil.diag45[1] + hmid * stencil.diag45[2];
                diag135[phase] = (ur - dl) * stencil.diag135[0] - vmid * stencil.diag135[1] + hmid * stencil.diag135[2];
            }

            in2[0]  = __floats2half2_rn(lr0.x, lr1.x);
            in2[1]  = __floats2half2_rn(lr2.x, lr3.x);
            in2[2]  = __floats2half2_rn(lr0.y, lr1.y);
            in2[3]  = __floats2half2_rn(lr2.y, lr3.y);
            in2[4]  = __floats2half2_rn(lr0.z, lr1.z);
            in2[5]  = __floats2half2_rn(lr2.z, lr3.z);
            in2[6]  = __floats2half2_rn(ly0,   ly1);
            in2[7]  = __floats2half2_rn(ly2,   ly3);
            in2[8]  = __floats2half2_rn(dx[0], dx[1]);
            in2[9]  = __floats2half2_rn(dx[2], dx[3]);
            in2[10] = __floats2half2_rn(dy[0], dy[1]);
            in2[11] = __floats2half2_rn(dy[2], dy[3]);
            in2[12] = __floats2half2_rn(diag45[0], diag45[1]);
            in2[13] = __floats2half2_rn(diag45[2], diag45[3]);
            in2[14] = __floats2half2_rn(diag135[0], diag135[1]);
            in2[15] = __floats2half2_rn(diag135[2], diag135[3]);
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 16; ++i) in2[i] = __float2half2_rn(0.0f);
        }

        const int pixel = py * packedWidth + px;

        if (activePixel && packedOutput != nullptr)
        {
            uint4* destination = reinterpret_cast<uint4*>(packedOutput) + size_t(pixel) * 4;
            #pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                destination[g] = make_uint4(
                    *reinterpret_cast<const uint32_t*>(&in2[g * 4 + 0]),
                    *reinterpret_cast<const uint32_t*>(&in2[g * 4 + 1]),
                    *reinterpret_cast<const uint32_t*>(&in2[g * 4 + 2]),
                    *reinterpret_cast<const uint32_t*>(&in2[g * 4 + 3])
                );
            }
        }

        const int warpId = tid / 32;
        const int laneId = tid % 32;

        half* s_act = reinterpret_cast<half*>(&s_out[warpId][0][0]);
        *reinterpret_cast<uint4*>(&s_act[laneId * 32 + 0])  = *reinterpret_cast<const uint4*>(&in2[0]);
        *reinterpret_cast<uint4*>(&s_act[laneId * 32 + 8])  = *reinterpret_cast<const uint4*>(&in2[4]);
        *reinterpret_cast<uint4*>(&s_act[laneId * 32 + 16]) = *reinterpret_cast<const uint4*>(&in2[8]);
        *reinterpret_cast<uint4*>(&s_act[laneId * 32 + 24]) = *reinterpret_cast<const uint4*>(&in2[12]);

        __syncwarp();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

        #pragma unroll
        for (int m = 0; m < 2; ++m)
        {
            #pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                nvcuda::wmma::fill_fragment(c_frag[m][n], 0.0f);
                #pragma unroll
                for (int k = 0; k < 2; ++k)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag, &s_act[(m * 16) * 32 + k * 16], 32);
                    nvcuda::wmma::load_matrix_sync(b_frag, projWeights + (n * 16) * 32 + k * 16, 32);
                    nvcuda::wmma::mma_sync(c_frag[m][n], a_frag, b_frag, c_frag[m][n]);
                }
            }
        }

        __syncwarp();

        #pragma unroll
        for (int m = 0; m < 2; ++m)
        {
            #pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                nvcuda::wmma::store_matrix_sync(&s_out[warpId][m * 16][n * 16], c_frag[m][n], 32, nvcuda::wmma::mem_row_major);
            }
        }

        __syncwarp();

        if (!activePixel) return;

        half outHalf[32];
        #pragma unroll
        for (int c = 0; c < 32; ++c)
        {
            float sum = s_out[warpId][laneId][c] + s_projBiases[c];
            float alpha = s_projPrelus[c];
            float act = sum > 0.0f ? sum : (sum * alpha);
            outHalf[c] = __float2half_rn(act);
        }

        uint4* dstSkip = reinterpret_cast<uint4*>(skipOutput) + size_t(pixel) * 4;
        const uint4* out_u4 = reinterpret_cast<const uint4*>(outHalf);
        dstSkip[0] = out_u4[0];
        dstSkip[1] = out_u4[1];
        dstSkip[2] = out_u4[2];
        dstSkip[3] = out_u4[3];
    }


    struct SharedExecutionEngine
    {
        struct PlanInfo
        {
            int height = 0;
            int width = 0;
            int inputChannels = 0;
            int outputChannels = 0;
            int filter = 1;
            int padding = 0;
            size_t outputElements = 0;
            Fp8ConvOperation fp8{};
            HalfConvOperation fp16{};
        };

        explicit SharedExecutionEngine(const std::filesystem::path& path)
            : model(ModelArtifact::Load(path)),
              easuKernelMode(ReadEasuKernelMode()),
              hasNvFp4Bottlenecks(model.HasNvFp4()),
              hasParallel32x4Bottlenecks(model.AdaptiveParallel32x4()),
              hasNativeFp8Bottlenecks(!model.HasNvFp4() && !model.AdaptiveParallel32x4() && model.Layers().size() > 3 &&
                  model.Layers()[2].weight.data != nullptr && model.Layers()[2].weight.byteCount == 36864 &&
                  model.Layers()[3].weight.data != nullptr && model.Layers()[3].weight.byteCount == 36864),
              hasFusedNativeFp8Expand(model.Layers().size() > (model.AdaptiveParallel32x4() ? 6u : 4u) &&
                  model.Layers()[model.AdaptiveParallel32x4() ? 6u : 4u].weight.data != nullptr &&
                  model.Layers()[model.AdaptiveParallel32x4() ? 6u : 4u].weight.byteCount == 8192 &&
                  model.Channels540() == 32),
              hasFusedNativeFp8Compress(model.Layers().size() > 1 &&
                  model.Layers()[1].weight.data != nullptr && model.Layers()[1].weight.byteCount == 8192 &&
                  model.Channels540() == 32)
        {
        }

        ~SharedExecutionEngine()
        {
            if (cudaDevice >= 0) cudaSetDevice(cudaDevice);
            if (stream != nullptr) cudaStreamSynchronize(stream);
            projectCudnn.reset();
            compressCudnn.reset();
            bottleneckCudnn.reset();
            expandCudnn.reset();
            refineCudnn.reset();
            tailCudnn.reset();
            DestroyCublasLt();
            if (cublasLt != nullptr) { cublasLtDestroy(cublasLt); cublasLt = nullptr; }
            if (d3dContext4 != nullptr) d3dContext4->Release();
            if (d3dDeviceIdentity != nullptr) d3dDeviceIdentity->Release();
            if (stream != nullptr) cudaStreamDestroy(stream);
        }

        void DestroyCublasLt() noexcept
        {
            if (tailPref != nullptr) { cublasLtMatmulPreferenceDestroy(tailPref); tailPref = nullptr; }
            if (tailDdesc != nullptr) { cublasLtMatrixLayoutDestroy(tailDdesc); tailDdesc = nullptr; }
            if (tailCdesc != nullptr) { cublasLtMatrixLayoutDestroy(tailCdesc); tailCdesc = nullptr; }
            if (tailBdesc != nullptr) { cublasLtMatrixLayoutDestroy(tailBdesc); tailBdesc = nullptr; }
            if (tailAdesc != nullptr) { cublasLtMatrixLayoutDestroy(tailAdesc); tailAdesc = nullptr; }
            if (tailMatmulDesc != nullptr) { cublasLtMatmulDescDestroy(tailMatmulDesc); tailMatmulDesc = nullptr; }
            hasTailCublasLt = false;
        }

        void ConfigureTailCublasLt(uint32_t pWidth, uint32_t pHeight)
        {
            DestroyCublasLt();
            if (cublasLt == nullptr)
            {
                CheckCublas(cublasLtCreate(&cublasLt), "cublasLtCreate");
            }
            cublasWorkspace.Allocate(4 * 1024 * 1024);
            const int64_t M = int64_t(pWidth) * pHeight;
            const int64_t K = 64;
            const int64_t N = 64;

            CheckCublas(cublasLtMatmulDescCreate(&tailMatmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                "cublasLtMatmulDescCreate(tail)");
            cublasOperation_t transA = CUBLAS_OP_T;
            cublasOperation_t transB = CUBLAS_OP_N;
            CheckCublas(cublasLtMatmulDescSetAttribute(tailMatmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)),
                "cublasLtMatmulDescSetAttribute(tail transA)");
            CheckCublas(cublasLtMatmulDescSetAttribute(tailMatmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)),
                "cublasLtMatmulDescSetAttribute(tail transB)");
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            CheckCublas(cublasLtMatmulDescSetAttribute(tailMatmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
                "cublasLtMatmulDescSetAttribute(tail epilogue)");
            void* biasPtr = tailBiasFp16.Get();
            CheckCublas(cublasLtMatmulDescSetAttribute(tailMatmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr)),
                "cublasLtMatmulDescSetAttribute(tail bias)");

            CheckCublas(cublasLtMatrixLayoutCreate(&tailAdesc, CUDA_R_16F, K, N, K), "cublasLtMatrixLayoutCreate(tailAdesc)");
            CheckCublas(cublasLtMatrixLayoutCreate(&tailBdesc, CUDA_R_16F, K, M, K), "cublasLtMatrixLayoutCreate(tailBdesc)");
            CheckCublas(cublasLtMatrixLayoutCreate(&tailCdesc, CUDA_R_16F, N, M, N), "cublasLtMatrixLayoutCreate(tailCdesc)");
            CheckCublas(cublasLtMatrixLayoutCreate(&tailDdesc, CUDA_R_16F, N, M, N), "cublasLtMatrixLayoutCreate(tailDdesc)");

            CheckCublas(cublasLtMatmulPreferenceCreate(&tailPref), "cublasLtMatmulPreferenceCreate(tailPref)");
            size_t wsSize = cublasWorkspace.Bytes();
            CheckCublas(cublasLtMatmulPreferenceSetAttribute(tailPref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)),
                "cublasLtMatmulPreferenceSetAttribute(tail wsSize)");

            constexpr int kMaxAlgos = 16;
            cublasLtMatmulHeuristicResult_t heuristicResults[kMaxAlgos];
            int returnedResults = 0;
            CheckCublas(cublasLtMatmulAlgoGetHeuristic(cublasLt, tailMatmulDesc, tailAdesc, tailBdesc, tailCdesc, tailDdesc, tailPref, kMaxAlgos, heuristicResults, &returnedResults),
                "cublasLtMatmulAlgoGetHeuristic(tail)");
            if (returnedResults == 0)
                throw std::runtime_error("No cuBLASLt heuristic algorithm found for tail");

            const size_t tailLayerIdx = model.Layers().size() - 1;
            int bestIdx = 0;
            if (weightsFp16[tailLayerIdx].Get() != nullptr && workAFp16.Get() != nullptr && tailFp16.Get() != nullptr && stream != nullptr)
            {
                float alpha = 1.0f, beta = 0.0f;
                float bestMs = 1e9f;
                cudaEvent_t evStart = nullptr, evStop = nullptr;
                if (cudaEventCreate(&evStart) == cudaSuccess && cudaEventCreate(&evStop) == cudaSuccess)
                {
                    for (int i = 0; i < returnedResults; ++i)
                    {
                        if (heuristicResults[i].state != CUBLAS_STATUS_SUCCESS) continue;
                        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
                        for (int w = 0; w < 2; ++w)
                        {
                            status = cublasLtMatmul(
                                cublasLt, tailMatmulDesc, &alpha,
                                weightsFp16[tailLayerIdx].Get(), tailAdesc,
                                workAFp16.Get(), tailBdesc,
                                &beta, nullptr, tailCdesc,
                                tailFp16.Get(), tailDdesc,
                                &heuristicResults[i].algo,
                                cublasWorkspace.Get(), cublasWorkspace.Bytes(),
                                stream);
                            if (status != CUBLAS_STATUS_SUCCESS) break;
                        }
                        if (status != CUBLAS_STATUS_SUCCESS) continue;
                        cudaStreamSynchronize(stream);

                        constexpr int kIters = 5;
                        cudaEventRecord(evStart, stream);
                        for (int it = 0; it < kIters; ++it)
                        {
                            status = cublasLtMatmul(
                                cublasLt, tailMatmulDesc, &alpha,
                                weightsFp16[tailLayerIdx].Get(), tailAdesc,
                                workAFp16.Get(), tailBdesc,
                                &beta, nullptr, tailCdesc,
                                tailFp16.Get(), tailDdesc,
                                &heuristicResults[i].algo,
                                cublasWorkspace.Get(), cublasWorkspace.Bytes(),
                                stream);
                            if (status != CUBLAS_STATUS_SUCCESS) break;
                        }
                        cudaEventRecord(evStop, stream);
                        cudaStreamSynchronize(stream);
                        if (status == CUBLAS_STATUS_SUCCESS)
                        {
                            float elapsedMs = 0.0f;
                            cudaEventElapsedTime(&elapsedMs, evStart, evStop);
                            if (elapsedMs < bestMs)
                            {
                                bestMs = elapsedMs;
                                bestIdx = i;
                            }
                        }
                    }
                    cudaEventDestroy(evStart);
                    cudaEventDestroy(evStop);
                }
            }
            tailHeuristicResult = heuristicResults[bestIdx];
            hasTailCublasLt = true;
        }

        void EnsureCudaDevice(ID3D11Device* device)
        {
            IDXGIDevice* dxgi = nullptr;
            HRESULT hr = device->QueryInterface(
                __uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgi));
            if (FAILED(hr) || dxgi == nullptr)
                throw std::runtime_error("SourcePyramid could not query IDXGIDevice");
            IDXGIAdapter* adapter = nullptr;
            hr = dxgi->GetAdapter(&adapter);
            dxgi->Release();
            if (FAILED(hr) || adapter == nullptr)
                throw std::runtime_error("SourcePyramid could not query the DXGI adapter");
            int selected = -1;
            const cudaError_t status = cudaD3D11GetDevice(&selected, adapter);
            adapter->Release();
            CheckCuda(status, "cudaD3D11GetDevice(SourcePyramid)");
            if (cudaDevice >= 0 && cudaDevice != selected)
                throw std::runtime_error("SourcePyramid execution engine cannot span CUDA devices");
            cudaDevice = selected;
            CheckCuda(cudaSetDevice(cudaDevice), "cudaSetDevice(SourcePyramid)");
            if (stream == nullptr)
            {
                int lowest = 0;
                int highest = 0;
                CheckCuda(
                    cudaDeviceGetStreamPriorityRange(&lowest, &highest),
                    "cudaDeviceGetStreamPriorityRange(SourcePyramid)");
                streamPriority = highest;
                CheckCuda(
                    cudaStreamCreateWithPriority(
                        &stream, cudaStreamNonBlocking, streamPriority),
                    "cudaStreamCreateWithPriority(SourcePyramid)");
                UploadModel();
            }
        }

        void EnsureD3DInterop(ID3D11Device* device)
        {
            IUnknown* identity = nullptr;
            HRESULT hr = device->QueryInterface(
                __uuidof(IUnknown), reinterpret_cast<void**>(&identity));
            if (FAILED(hr) || identity == nullptr)
                throw std::runtime_error("SourcePyramid could not identify its D3D11 device");
            if (d3dDeviceIdentity != nullptr)
            {
                const bool same = d3dDeviceIdentity == identity;
                identity->Release();
                if (!same)
                    throw std::runtime_error("SourcePyramid sibling slots require one D3D11 device");
                return;
            }
            ID3D11DeviceContext* context = nullptr;
            device->GetImmediateContext(&context);
            ID3D11DeviceContext4* context4 = nullptr;
            hr = context->QueryInterface(
                __uuidof(ID3D11DeviceContext4), reinterpret_cast<void**>(&context4));
            context->Release();
            if (FAILED(hr) || context4 == nullptr)
            {
                identity->Release();
                throw std::runtime_error("SourcePyramid requires ID3D11DeviceContext4");
            }
            d3dDeviceIdentity = identity;
            d3dContext4 = context4;
        }

        template <typename T>
        void UploadWeight(const TensorView& view, DeviceBuffer& destination,
            uint32_t minimumInputChannels=0,uint32_t minimumOutputChannels=0)
        {
            const uint32_t outputChannels = view.shape[0];
            const uint32_t inputChannels = view.shape[1];
            const uint32_t rows = view.shape[2];
            const uint32_t columns = view.shape[3];
            const uint32_t storedInput=std::max(inputChannels,minimumInputChannels);
            const uint32_t storedOutput=std::max(outputChannels,minimumOutputChannels);
            std::vector<T> krsc(
                size_t(storedOutput) * rows * columns * storedInput);
            const T* oihw = reinterpret_cast<const T*>(view.data);
            for (uint32_t output = 0; output < outputChannels; ++output)
                for (uint32_t row = 0; row < rows; ++row)
                    for (uint32_t column = 0; column < columns; ++column)
                        for (uint32_t input = 0; input < inputChannels; ++input)
                            krsc[((size_t(output) * rows + row) * columns + column) * storedInput + input] =
                                oihw[((size_t(output) * inputChannels + input) * rows + row) * columns + column];
            destination.Allocate(krsc.size() * sizeof(T));
            CheckCuda(
                cudaMemcpy(
                    destination.Get(), krsc.data(), destination.Bytes(),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(SourcePyramid weight)");
        }

        void UploadWeightNvFp4(const LayerView& layer, DeviceBuffer& destinationFp8, DeviceBuffer& destinationFp16,
            const float* channelScales)
        {
            const auto& view = layer.weightNvFp4;
            const uint32_t outputChannels = view.shape[0];
            const uint32_t inputChannels = view.shape[1];
            const uint32_t rows = view.shape[2];
            const uint32_t columns = view.shape[3];
            const size_t totalElements = size_t(outputChannels) * inputChannels * rows * columns;

            const uint8_t* packedBytes = reinterpret_cast<const uint8_t*>(view.data);
            std::vector<float> oihwFloat(totalElements);
            const bool isPerChannel = (layer.weightNvFp4Scale.dataType == TensorDataType::Fp32);
            const float* chScalesFloat = reinterpret_cast<const float*>(layer.weightNvFp4Scale.data);
            const uint16_t* blockScalesHalf = reinterpret_cast<const uint16_t*>(layer.weightNvFp4Scale.data);
            constexpr size_t blockSize = 16;
            const size_t elementsPerOutChannel = inputChannels * rows * columns;

            static constexpr float kE2M1Lut[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
            };

            for (size_t i = 0; i < totalElements; ++i)
            {
                const uint8_t b = packedBytes[i / 2];
                const uint8_t nibble = (i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                const float val = kE2M1Lut[nibble];
                float scale = 1.0f;
                if (isPerChannel)
                {
                    const size_t outIdx = i / elementsPerOutChannel;
                    scale = chScalesFloat[outIdx];
                }
                else
                {
                    const size_t blockIdx = i / blockSize;
                    scale = __half2float(*reinterpret_cast<const __half*>(&blockScalesHalf[blockIdx]));
                }
                oihwFloat[i] = val * scale;
            }

            std::vector<uint8_t> krscFp8(totalElements);
            std::vector<uint16_t> krscFp16(totalElements);
            for (uint32_t output = 0; output < outputChannels; ++output)
            {
                const float chScale = channelScales[output];
                const float invScale = (chScale > 1e-12f) ? (1.0f / chScale) : 1.0f;
                for (uint32_t row = 0; row < rows; ++row)
                {
                    for (uint32_t column = 0; column < columns; ++column)
                    {
                        for (uint32_t input = 0; input < inputChannels; ++input)
                        {
                            const size_t oihwIdx = ((size_t(output) * inputChannels + input) * rows + row) * columns + column;
                            const size_t krscIdx = ((size_t(output) * rows + row) * columns + column) * inputChannels + input;
                            const float w = oihwFloat[oihwIdx];
                            const __nv_fp8_e4m3 fp8Val(w * invScale);
                            krscFp8[krscIdx] = *reinterpret_cast<const uint8_t*>(&fp8Val);
                            const __half halfVal(w);
                            krscFp16[krscIdx] = *reinterpret_cast<const uint16_t*>(&halfVal);
                        }
                    }
                }
            }
            destinationFp8.Allocate(krscFp8.size() * sizeof(uint8_t));
            CheckCuda(cudaMemcpy(destinationFp8.Get(), krscFp8.data(), destinationFp8.Bytes(), cudaMemcpyHostToDevice),
                "cudaMemcpy(NVFP4 dequantized weight FP8)");
            destinationFp16.Allocate(krscFp16.size() * sizeof(uint16_t));
            CheckCuda(cudaMemcpy(destinationFp16.Get(), krscFp16.data(), destinationFp16.Bytes(), cudaMemcpyHostToDevice),
                "cudaMemcpy(NVFP4 dequantized weight FP16)");
        }

        void UploadModel()
        {
            if (model.AdaptiveFilter())
            {
                const auto& basis=model.AdaptiveBasis();
                if (model.AdaptivePairMoE())
                {
                    std::memcpy(adaptiveLearnedMoEBank.values,basis.data,sizeof(adaptiveLearnedMoEBank.values));
                    std::memcpy(adaptiveLearnedMoEBank.base,model.AdaptiveBase().data,sizeof(adaptiveLearnedMoEBank.base));
                    routeThresholdDevice.Allocate(64 * sizeof(float));
                    CheckCuda(cudaMemcpy(routeThresholdDevice.Get(),model.RouteThreshold().data,
                        64 * sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy(route_threshold)");
                }
                else
                {
                    std::memcpy(&adaptiveBasisBank,basis.data,sizeof(adaptiveBasisBank));
                    if (model.AdaptiveOc25())
                    {
                        std::memcpy(adaptiveLearnedBank.values,basis.data,sizeof(adaptiveLearnedBank.values));
                        std::memcpy(adaptiveLearnedBank.base,model.AdaptiveBase().data,sizeof(adaptiveLearnedBank.base));
                        for (int i = 0; i < 25 * 16; ++i)
                        {
                            const float4 v = adaptiveLearnedBank.values[i];
                            adaptiveLearnedBankHalf.values[i].xy = make_half2(__float2half(v.x), __float2half(v.y));
                            adaptiveLearnedBankHalf.values[i].zw = make_half2(__float2half(v.z), __float2half(v.w));
                        }
                        for (int i = 0; i < 25; ++i)
                        {
                            const float4 v = adaptiveLearnedBank.base[i];
                            adaptiveLearnedBankHalf.base[i].xy = make_half2(__float2half(v.x), __float2half(v.y));
                            adaptiveLearnedBankHalf.base[i].zw = make_half2(__float2half(v.z), __float2half(v.w));
                        }
                    }
                }
            }
            for (size_t layer = 0; layer < model.Layers().size(); ++layer)
            {
                const LayerView& value = model.Layers()[layer];
                const bool padProject=model.AdaptiveFilter()&&!model.AdaptiveFeature8()&&!model.AdaptivePairMoE()&&!model.Fp8HalfProject();
                if (value.hasNvFp4)
                {
                    const float* chScales = reinterpret_cast<const float*>(value.weightScale.data);
                    UploadWeightNvFp4(value, weightsFp8[layer], weightsFp16[layer], chScales);
                }
                else
                {
                    UploadWeight<uint8_t>(value.weight, weightsFp8[layer],
                        (padProject&&layer==0) ? 32u : 0u, 0u);
                    UploadWeight<uint16_t>(value.weightFp16, weightsFp16[layer]);
                }
                weightScales[layer].Allocate(value.weightScale.byteCount);
                biases[layer].Allocate(value.bias.byteCount);
                if (model.AdaptiveFilter())
                {
                    biasesFp8[layer].Allocate(value.biasFp8.byteCount);
                    CheckCuda(cudaMemcpy(biasesFp8[layer].Get(),value.biasFp8.data,
                        value.biasFp8.byteCount,cudaMemcpyHostToDevice),"upload calibrated FP8 bias");
                }
                prelus[layer].Allocate(value.prelu.byteCount);
                CheckCuda(cudaMemcpy(
                    weightScales[layer].Get(), value.weightScale.data,
                    value.weightScale.byteCount, cudaMemcpyHostToDevice),
                    "cudaMemcpy(SourcePyramid weight scale)");
                CheckCuda(cudaMemcpy(
                    biases[layer].Get(), value.bias.data, value.bias.byteCount,
                    cudaMemcpyHostToDevice),
                    "cudaMemcpy(SourcePyramid bias)");
                CheckCuda(cudaMemcpy(
                    prelus[layer].Get(), value.prelu.data, value.prelu.byteCount,
                    cudaMemcpyHostToDevice),
                    "cudaMemcpy(SourcePyramid PReLU)");
            }
            const size_t tailLayerIdx = model.Layers().size() - 1;
            const size_t refineLayerIdx = model.Layers().size() - 2;
            const size_t expandLayerIdx = hasParallel32x4Bottlenecks ? 6 : 4;
            if ((model.AdaptivePairMoE() || model.AdaptiveFilter()) && model.Layers().size() > tailLayerIdx)
            {
                const float* bFloat = reinterpret_cast<const float*>(model.Layers()[tailLayerIdx].bias.data);
                std::vector<half> bHalf(64);
                for (size_t i = 0; i < 64; ++i) bHalf[i] = __float2half(bFloat[i]);
                tailBiasFp16.Allocate(64 * sizeof(half));
                CheckCuda(cudaMemcpy(tailBiasFp16.Get(), bHalf.data(), 64 * sizeof(half), cudaMemcpyHostToDevice),
                    "cudaMemcpy(tailBiasFp16)");
            }
            if (model.AdaptiveFilter() && !model.AdaptivePairMoE() && model.Layers().size() > refineLayerIdx)
            {
                const LayerView& refineLayer = model.Layers()[refineLayerIdx];
                if (useNativeFp16Refine && refineLayer.weightFp16.data != nullptr &&
                    refineLayer.weightFp16.byteCount == 36864)
                {
                    const half* wRefine = reinterpret_cast<const half*>(refineLayer.weightFp16.data);
                    std::vector<uint4> packedW(kNativeFp16RefinePackedU4Count);
                    PackNativeFp16RefineWeights(wRefine, packedW.data());
                    refineWeightsPacked.Allocate(kNativeFp16RefinePackedBytes);
                    CheckCuda(cudaMemcpy(refineWeightsPacked.Get(), packedW.data(),
                        kNativeFp16RefinePackedBytes, cudaMemcpyHostToDevice),
                        "cudaMemcpy(refineWeightsPacked)");
                    hasFastWmmaRefine = true;

                    if (useFusedTail && model.Layers().size() > tailLayerIdx)
                    {
                        const LayerView& tailLayer = model.Layers()[tailLayerIdx];
                        if (tailLayer.weightFp16.data != nullptr && tailLayer.weightFp16.byteCount == 8192)
                        {
                            const half* wTail = reinterpret_cast<const half*>(tailLayer.weightFp16.data);
                            std::vector<uint4> packedTail(kNativeFp16TailPackedU4Count);
                            PackNativeFp16TailWeights(wTail, packedTail.data());
                            tailWeightsPacked.Allocate(kNativeFp16TailPackedBytes);
                            CheckCuda(cudaMemcpy(tailWeightsPacked.Get(), packedTail.data(),
                                kNativeFp16TailPackedBytes, cudaMemcpyHostToDevice),
                                "cudaMemcpy(tailWeightsPacked)");
                            hasFusedRefineTail = true;
                        }
                    }
                }
            }
            hasNvFp4Bottlenecks = model.Layers().size() > 3 &&
                model.Layers()[2].hasNvFp4 && model.Layers()[3].hasNvFp4;
            if (hasNvFp4Bottlenecks)
            {
                std::vector<uint4> packedMma(kBlackwellFp4PackedU4Count);
                for (size_t bIdx = 0; bIdx < 2; ++bIdx)
                {
                    const LayerView& value = model.Layers()[bIdx + 2];
                    PackBlackwellFp4BottleneckWeights(
                        reinterpret_cast<const uint8_t*>(value.weightNvFp4.data),
                        packedMma.data());
                    weightsNvFp4Mma[bIdx].Allocate(kBlackwellFp4PackedBytes);
                    CheckCuda(
                        cudaMemcpy(weightsNvFp4Mma[bIdx].Get(), packedMma.data(),
                                   kBlackwellFp4PackedBytes, cudaMemcpyHostToDevice),
                        "cudaMemcpy(Blackwell FP4 MMA weights)");
                    weightsNvFp4Scale[bIdx].Allocate(value.weightNvFp4Scale.byteCount);
                    CheckCuda(
                        cudaMemcpy(weightsNvFp4Scale[bIdx].Get(), value.weightNvFp4Scale.data,
                                   value.weightNvFp4Scale.byteCount, cudaMemcpyHostToDevice),
                        "cudaMemcpy(Blackwell FP4 MMA scales)");
                }
            }
            if (hasParallel32x4Bottlenecks)
            {
                std::vector<uint4> packedMma(kNativeFp8Grouped64PackedU4Count);
                std::vector<half> packedMmaFp16(kNativeFp16Grouped64PackedHalfs);
                for (size_t bIdx = 0; bIdx < 4; ++bIdx)
                {
                    const LayerView& value = model.Layers()[bIdx + 2];
                    PackNativeFp8GroupedWeights(
                        reinterpret_cast<const uint8_t*>(value.weight.data),
                        packedMma.data());
                    weightsFp8Mma[bIdx].Allocate(kNativeFp8Grouped64PackedBytes);
                    CheckCuda(
                        cudaMemcpy(weightsFp8Mma[bIdx].Get(), packedMma.data(),
                                   kNativeFp8Grouped64PackedBytes, cudaMemcpyHostToDevice),
                        "cudaMemcpy(Native FP8 Grouped MMA weights)");

                    if (value.weightFp16.data != nullptr)
                    {
                        PackNativeFp16GroupedWeights(
                            reinterpret_cast<const half*>(value.weightFp16.data),
                            packedMmaFp16.data());
                        weightsFp16Mma[bIdx].Allocate(kNativeFp16Grouped64PackedBytes);
                        CheckCuda(
                            cudaMemcpy(weightsFp16Mma[bIdx].Get(), packedMmaFp16.data(),
                                       kNativeFp16Grouped64PackedBytes, cudaMemcpyHostToDevice),
                            "cudaMemcpy(Native FP16 Grouped MMA weights)");
                    }
                }
            }
            else if (hasNativeFp8Bottlenecks)
            {
                std::vector<uint4> packedMma(kNativeFp8PackedU4Count);
                for (size_t bIdx = 0; bIdx < 2; ++bIdx)
                {
                    const LayerView& value = model.Layers()[bIdx + 2];
                    PackNativeFp8BottleneckWeights(
                        reinterpret_cast<const uint8_t*>(value.weight.data),
                        packedMma.data());
                    weightsFp8Mma[bIdx].Allocate(kNativeFp8PackedBytes);
                    CheckCuda(
                        cudaMemcpy(weightsFp8Mma[bIdx].Get(), packedMma.data(),
                                   kNativeFp8PackedBytes, cudaMemcpyHostToDevice),
                        "cudaMemcpy(Native FP8 MMA weights)");
                }
            }
            if (hasFusedNativeFp8Expand)
            {
                std::vector<uint4> packedExpand(kNativeFp8ExpandPackedU4Count);
                PackNativeFp8ExpandWeights(
                    reinterpret_cast<const uint8_t*>(model.Layers()[expandLayerIdx].weight.data),
                    packedExpand.data());
                weightsFp8ExpandMma.Allocate(kNativeFp8ExpandPackedBytes);
                CheckCuda(
                    cudaMemcpy(weightsFp8ExpandMma.Get(), packedExpand.data(),
                               kNativeFp8ExpandPackedBytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy(Native FP8 Expand MMA weights)");
            }
            if (hasFusedNativeFp8Compress)
            {
                std::vector<uint4> packedCompress(kNativeFp8CompressPackedU4Count);
                PackNativeFp8CompressWeights(
                    reinterpret_cast<const uint8_t*>(model.Layers()[1].weight.data),
                    packedCompress.data());
                weightsFp8CompressMma.Allocate(kNativeFp8CompressPackedBytes);
                CheckCuda(
                    cudaMemcpy(weightsFp8CompressMma.Get(), packedCompress.data(),
                               kNativeFp8CompressPackedBytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy(Native FP8 Compress MMA weights)");
            }
            if (model.AdaptiveFilter() && !model.AdaptiveFeature8() && !model.Fp8HalfHead())
            {
                auto padVector=[&](const TensorView& input,DeviceBuffer& output,float fill)
                {
                    std::array<float,64> padded;
                    padded.fill(fill);
                    std::memcpy(padded.data(),input.data,input.byteCount);
                    output.Allocate(sizeof(padded));
                    CheckCuda(cudaMemcpy(output.Get(),padded.data(),sizeof(padded),cudaMemcpyHostToDevice),
                        "upload padded FP8 refine vector");
                };
                UploadWeight<uint8_t>(model.Layers()[5].weight, weightFp8Refine64, 0u, 64u);
                UploadWeight<uint8_t>(model.Layers()[6].weight, weightFp8Tail64, 64u, 0u);
                padVector(model.Layers()[5].weightScale, weightScaleRefine64, 1.0f);
                padVector(model.Layers()[5].bias, biasRefine64, 0.0f);
                padVector(model.Layers()[5].biasFp8, biasFp8Refine64, 0.0f);
                padVector(model.Layers()[5].prelu, preluRefine64, 0.1f);
            }
            const float* scales = reinterpret_cast<const float*>(
                model.ActivationScales().data);
            const size_t scaleCount = model.ActivationScales().shape[0];
            activationScales.resize(scaleCount);
            std::copy_n(scales, scaleCount, activationScales.begin());
            activationScalesDevice.Allocate(
                activationScales.size() * sizeof(float));
            CheckCuda(cudaMemcpy(
                activationScalesDevice.Get(),activationScales.data(),
                activationScalesDevice.Bytes(),cudaMemcpyHostToDevice),
                "cudaMemcpy(SourcePyramid activation scales)");
            std::vector<float> reciprocals(scaleCount);
            for (size_t index=0;index<scaleCount;++index)
                reciprocals[index]=1.0f/activationScales[index];
            outputReciprocalScalesDevice.Allocate(
                reciprocals.size()*sizeof(float));
            CheckCuda(cudaMemcpy(
                outputReciprocalScalesDevice.Get(),reciprocals.data(),
                outputReciprocalScalesDevice.Bytes(),cudaMemcpyHostToDevice),
                "cudaMemcpy(SourcePyramid reciprocal activation scales)");
        }

        void* Fp8InputPointer(size_t layer) const noexcept
        {
            switch (layer)
            {
            case 0: return packedFp8.Get();
            case 1: return workAFp8.Get();
            case 2: return lowAFp8.Get();
            case 3: return lowBFp8.Get();
            case 4: return lowAFp8.Get();
            case 5: return workBFp8.Get();
            case 6: return workAFp8.Get();
            default: return nullptr;
            }
        }

        void* Fp16InputPointer(size_t layer) const noexcept
        {
            switch (layer)
            {
            case 0: return packedFp16.Get();
            case 1: return workAFp16.Get();
            case 2: return lowAFp16.Get();
            case 3: return lowBFp16.Get();
            case 4: return lowAFp16.Get();
            case 5: return workBFp16.Get();
            case 6: return workAFp16.Get();
            default: return nullptr;
            }
        }

        cutlass::conv::Conv2dProblemSize Problem(size_t layer) const
        {
            const PlanInfo& value = plans[layer];
            cutlass::Tensor4DCoord input(
                1, value.height, value.width, value.inputChannels);
            cutlass::Tensor4DCoord filter(
                value.outputChannels, value.filter, value.filter,
                value.inputChannels);
            cutlass::Tensor4DCoord output(
                1, value.height, value.width, value.outputChannels);
            return cutlass::conv::Conv2dProblemSize(
                input, filter,
                {value.padding,value.padding,value.padding,value.padding},
                {1,1}, {1,1}, output,
                cutlass::conv::Mode::kCrossCorrelation, 1);
        }

        typename Fp8ConvOperation::Arguments Fp8Arguments(size_t layer) const
        {
            const PlanInfo& value = plans[layer];
            cutlass::Tensor4DCoord input(
                1,value.height,value.width,value.inputChannels);
            cutlass::Tensor4DCoord filter(
                value.outputChannels,value.filter,value.filter,value.inputChannels);
            cutlass::Tensor4DCoord output(
                1,value.height,value.width,value.outputChannels);
            return typename Fp8ConvOperation::Arguments{
                Problem(layer),
                {reinterpret_cast<Fp8*>(Fp8InputPointer(layer)),Nhwc::packed(input)},
                {weightsFp8[layer].As<Fp8>(),Nhwc::packed(filter)},
                {accumulator.As<float>(),Nhwc::packed(output)},
                {accumulator.As<float>(),Nhwc::packed(output)},
                {1.0f,0.0f}};
        }

        typename HalfConvOperation::Arguments Fp16Arguments(size_t layer) const
        {
            const PlanInfo& value = plans[layer];
            cutlass::Tensor4DCoord input(
                1,value.height,value.width,value.inputChannels);
            cutlass::Tensor4DCoord filter(
                value.outputChannels,value.filter,value.filter,value.inputChannels);
            cutlass::Tensor4DCoord output(
                1,value.height,value.width,value.outputChannels);
            return typename HalfConvOperation::Arguments{
                Problem(layer),
                {reinterpret_cast<Half*>(Fp16InputPointer(layer)),Nhwc::packed(input)},
                {weightsFp16[layer].As<Half>(),Nhwc::packed(filter)},
                {accumulator.As<float>(),Nhwc::packed(output)},
                {accumulator.As<float>(),Nhwc::packed(output)},
                {1.0f,0.0f}};
        }

        void InitializePlans(RuntimePrecision precision)
        {
            const auto usesCutlass = [precision](size_t layer) noexcept
            {
                (void)layer;
                return precision == RuntimePrecision::Fp16;
            };
            size_t workspaceBytes = 0;
            for (size_t layer = 0; layer < 7; ++layer)
            {
                if (!usesCutlass(layer)) continue;
                if (precision == RuntimePrecision::Fp8)
                {
                    auto arguments = Fp8Arguments(layer);
                    CheckCutlass(
                        Fp8ConvOperation::can_implement(arguments),
                        "SourcePyramid CUTLASS FP8 can_implement");
                    workspaceBytes = std::max(
                        workspaceBytes,
                        plans[layer].fp8.get_workspace_size(arguments));
                }
                else
                {
                    auto arguments = Fp16Arguments(layer);
                    CheckCutlass(
                        HalfConvOperation::can_implement(arguments),
                        "SourcePyramid CUTLASS FP16 can_implement");
                    workspaceBytes = std::max(
                        workspaceBytes,
                        plans[layer].fp16.get_workspace_size(arguments));
                }
            }
            workspace.Allocate(workspaceBytes);
            for (size_t layer = 0; layer < 7; ++layer)
            {
                if (!usesCutlass(layer)) continue;
                if (precision == RuntimePrecision::Fp8)
                {
                    auto arguments = Fp8Arguments(layer);
                    CheckCutlass(
                        plans[layer].fp8.initialize(
                            arguments, workspace.Get(), stream),
                        "SourcePyramid CUTLASS FP8 initialize");
                }
                else
                {
                    auto arguments = Fp16Arguments(layer);
                    CheckCutlass(
                        plans[layer].fp16.initialize(
                            arguments, workspace.Get(), stream),
                        "SourcePyramid CUTLASS FP16 initialize");
                }
            }
        }

        void AttachConfiguration(
            uint32_t width, uint32_t height, RuntimePrecision precision)
        {
            if (configurationValid && configuredWidth == width &&
                configuredHeight == height && configuredPrecision == precision)
            {
                ++configuredSlotCount;
                return;
            }
            if (configuredSlotCount != 0)
                throw std::runtime_error(
                    "SourcePyramid sibling slots require identical dimensions and precision");
            if (width > std::numeric_limits<uint32_t>::max() / 2 ||
                height > std::numeric_limits<uint32_t>::max() / 2)
                throw std::runtime_error("SourcePyramid dimensions overflow exact-2x output");

            configurationValid = false;
            configuredSlotCount = 0;
            configuredWidth = configuredHeight = 0;
            startupReport.clear();
            persistentBytes = 0;
            try
            {
                CheckCuda(
                    cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(SourcePyramid configure)");
                projectCudnn.reset();
                compressCudnn.reset();
                bottleneckCudnn.reset();
                expandCudnn.reset();
                refineCudnn.reset();
                tailCudnn.reset();
                const bool direct = model.AdaptiveFilter();
                const bool halfProject = model.Fp8HalfProject();
                const bool halfHead = model.Fp8HalfHead();
                paddedWidth = direct ? ((width + 31u) & ~31u) + 32u : (width + 19u) & ~3u;
                paddedHeight = direct ? ((height + 31u) & ~31u) + 32u : (height + 19u) & ~3u;
                sourcePadX = direct ? 16u + ((32u - width % 32u) % 32u) / 2u : 8u;
                sourcePadY = direct ? 16u + ((32u - height % 32u) % 32u) / 2u : 8u;
                const uint32_t packedChannels = model.PackedInputChannels();
                const uint32_t channels540 = model.Channels540();
                const uint32_t channels270In = channels540 * 4;
                const uint32_t channels270Out = channels540 * 4;
                const uint32_t refineChannelsBase = model.RefineChannels();
                const uint32_t tailChannels = model.TailChannels();
                packedWidth = paddedWidth / 2;
                packedHeight = paddedHeight / 2;
                lowWidth = packedWidth / 2;
                lowHeight = packedHeight / 2;
                const bool padProject = precision==RuntimePrecision::Fp8&&direct&&!halfProject&&!model.AdaptiveFeature8();
                const bool padRefine = precision==RuntimePrecision::Fp8&&direct&&!halfHead&&!model.AdaptiveFeature8();
                projectInputChannels = padProject ? 32u : packedChannels;
                refineChannels = padRefine ? 64u : refineChannelsBase;
                hasFusedProject = useFusedProject &&
                    model.AdaptiveLearnedRgb4Dir() &&
                    projectInputChannels == 32 &&
                    channels540 == 32 &&
                    (precision == RuntimePrecision::Mixed || precision == RuntimePrecision::Fp16) &&
                    model.Layers().size() > 0 &&
                    model.Layers()[0].weightFp16.data != nullptr &&
                    model.Layers()[0].weightFp16.byteCount == 2048;
                outputWidth = width * 2;
                outputHeight = height * 2;
                const size_t packedPixels = size_t(packedWidth) * packedHeight;
                const size_t lowPixels = size_t(lowWidth) * lowHeight;
                const size_t packed540 = packedPixels * channels540;
                const size_t packedRefine = packedPixels * refineChannels;
                const size_t low64 = lowPixels * 64;
                const size_t lowUnshuffle = lowPixels * channels270In;
                const size_t packedElements = packedPixels * projectInputChannels;

                auto setPlan = [this](
                    size_t index, int h, int w, int inChannels,
                    int outChannels, int filter, int padding)
                {
                    PlanInfo& value = plans[index];
                    value.height = h;
                    value.width = w;
                    value.inputChannels = inChannels;
                    value.outputChannels = outChannels;
                    value.filter = filter;
                    value.padding = padding;
                    value.outputElements = size_t(h) * w * outChannels;
                };
                setPlan(0,packedHeight,packedWidth,projectInputChannels,channels540,1,0);
                setPlan(1,lowHeight,lowWidth,channels270In,64,1,0);
                setPlan(2,lowHeight,lowWidth,64,64,3,1);
                setPlan(3,lowHeight,lowWidth,64,64,3,1);
                setPlan(4,lowHeight,lowWidth,64,channels270Out,1,0);
                setPlan(5,packedHeight,packedWidth,channels540,refineChannels,3,1);
                setPlan(6,packedHeight,packedWidth,refineChannels,tailChannels,1,0);

                edge.Allocate(direct ? 0 : size_t(outputWidth)*outputHeight*sizeof(ushort4));
                accumulator.Allocate(0);
                tailFp16.Allocate(packedPixels*tailChannels*sizeof(Half));
                routeMetaFp16.Allocate(model.AdaptivePairMoE() ? (packedPixels * 64 * sizeof(Half)) : 0);
                if (precision == RuntimePrecision::Fp8)
                {
                    packedFp8.Allocate(halfProject ? 0 : packedElements);
                    skipFp8.Allocate(packed540);
                    workAFp8.Allocate(std::max(packedRefine, lowUnshuffle));
                    workBFp8.Allocate(packed540);
                    lowAFp8.Allocate(low64);
                    lowBFp8.Allocate(low64);
                    // Input packing dies after projection. Reuse the same
                    // storage for the FP16 refine output consumed by the head.
                    packedFp16.Allocate(halfHead ? packedRefine*sizeof(Half) : (halfProject ? packedElements*sizeof(Half) : 0));
                    skipFp16.Allocate(0);
                    workAFp16.Allocate(0); workBFp16.Allocate(0);
                    lowAFp16.Allocate(0); lowBFp16.Allocate(0);
                }
                else if (precision == RuntimePrecision::Mixed)
                {
                    packedFp16.Allocate(packedElements * sizeof(Half));
                    skipFp16.Allocate(packed540 * sizeof(Half));
                    workAFp8.Allocate(std::max(packedRefine, lowUnshuffle));
                    lowAFp8.Allocate(low64);
                    lowBFp8.Allocate(low64);
                    workBFp16.Allocate(packed540 * sizeof(Half));
                    workAFp16.Allocate(packedRefine * sizeof(Half));
                    packedFp8.Allocate(0); skipFp8.Allocate(0);
                    workBFp8.Allocate(0);
                    lowAFp16.Allocate(0); lowBFp16.Allocate(0);
                }
                else
                {
                    packedFp16.Allocate(packedElements * sizeof(Half));
                    skipFp16.Allocate(packed540 * sizeof(Half));
                    workAFp16.Allocate(std::max(packedRefine, lowUnshuffle) * sizeof(Half));
                    workBFp16.Allocate(packed540 * sizeof(Half));
                    lowAFp16.Allocate(low64 * sizeof(Half));
                    lowBFp16.Allocate(low64 * sizeof(Half));
                    packedFp8.Allocate(0); skipFp8.Allocate(0);
                    workAFp8.Allocate(0); workBFp8.Allocate(0);
                    lowAFp8.Allocate(0); lowBFp8.Allocate(0);
                }
                workspace.Allocate(0);

                if (precision == RuntimePrecision::Fp8)
                {
                    auto scale = [this](size_t index)
                    {
                        return activationScalesDevice.As<float>() + index;
                    };
                    auto reciprocal = [this](size_t index)
                    {
                        return outputReciprocalScalesDevice.As<float>() + index;
                    };
                    projectCudnn = std::make_unique<CudnnConvRuntime>();
                    projectCudnn->Configure(
                        {packedWidth,packedHeight,projectInputChannels,channels540,1,0,"project",
                         CudnnConvEpilogue::Fp8PreluRequant,
                         halfProject ? CudnnConvDataType::Fp16 : CudnnConvDataType::Fp8E4m3,
                         precision==RuntimePrecision::Fp8&&direct},stream,
                        {{halfProject ? packedFp16.Get() : packedFp8.Get(),
                          halfProject ? weightsFp16[0].Get() : weightsFp8[0].Get(),halfProject ? nullptr : scale(0),
                          halfProject ? nullptr : weightScales[0].As<float>(),Fp8Bias(0),
                          prelus[0].As<float>(),reciprocal(1),skipFp8.Get()}});
                    compressCudnn = std::make_unique<CudnnConvRuntime>();
                    compressCudnn->Configure(
                        {lowWidth,lowHeight,channels270In,64,1,0,"compress"},stream,
                        {{workAFp8.Get(),weightsFp8[1].Get(),scale(1),
                          weightScales[1].As<float>(),Fp8Bias(1),
                          prelus[1].As<float>(),reciprocal(2),lowAFp8.Get()}});
                    if (!hasParallel32x4Bottlenecks)
                    {
                        bottleneckCudnn = std::make_unique<CudnnConvRuntime>();
                        bottleneckCudnn->Configure(
                            {lowWidth,lowHeight,64,64,3,1,"bottleneck"},stream,
                            {
                                {lowAFp8.Get(),weightsFp8[2].Get(),scale(2),
                                 weightScales[2].As<float>(),Fp8Bias(2),
                                 prelus[2].As<float>(),reciprocal(3),lowBFp8.Get()},
                                {lowBFp8.Get(),weightsFp8[3].Get(),scale(3),
                                 weightScales[3].As<float>(),Fp8Bias(3),
                                 prelus[3].As<float>(),reciprocal(4),lowAFp8.Get()},
                            });
                    }
                    else
                    {
                        bottleneckCudnn.reset();
                    }
                    const size_t expandLayerIdx = hasParallel32x4Bottlenecks ? 6 : 4;
                    const size_t expandInActIdx = hasParallel32x4Bottlenecks ? 6 : 4;
                    const size_t expandOutActIdx = hasParallel32x4Bottlenecks ? 7 : 5;
                    expandCudnn = std::make_unique<CudnnConvRuntime>();
                    expandCudnn->Configure(
                        {lowWidth,lowHeight,64,channels270Out,1,0,"expand"},stream,
                        {{lowAFp8.Get(),weightsFp8[expandLayerIdx].Get(),scale(expandInActIdx),
                          weightScales[expandLayerIdx].As<float>(),Fp8Bias(expandLayerIdx),
                          prelus[expandLayerIdx].As<float>(),reciprocal(expandOutActIdx),workAFp8.Get()}});
                    const size_t refineLayerIdx = model.Layers().size() - 2;
                    const size_t tailLayerIdx = model.Layers().size() - 1;
                    const DeviceBuffer& curRefineWeight = padRefine ? weightFp8Refine64 : weightsFp8[refineLayerIdx];
                    const float* curRefineWeightScale = padRefine ? weightScaleRefine64.As<float>() : weightScales[refineLayerIdx].As<float>();
                    const float* curRefineBias = padRefine ? (halfHead ? biasRefine64.As<float>() : biasFp8Refine64.As<float>()) : Fp8Bias(refineLayerIdx);
                    const float* curRefinePrelu = padRefine ? preluRefine64.As<float>() : prelus[refineLayerIdx].As<float>();

                    refineCudnn = std::make_unique<CudnnConvRuntime>();
                    refineCudnn->Configure(
                        {packedWidth,packedHeight,channels540,refineChannels,3,1,"refine",
                         halfHead ? CudnnConvEpilogue::Fp16Prelu : CudnnConvEpilogue::Fp8PreluRequant,
                         CudnnConvDataType::Fp8E4m3,true},stream,
                        {{workBFp8.Get(),curRefineWeight.Get(),scale(expandOutActIdx + 1),
                          curRefineWeightScale,curRefineBias,
                          curRefinePrelu,halfHead ? nullptr : reciprocal(expandOutActIdx + 2),
                          halfHead ? packedFp16.Get() : workAFp8.Get()}});
                    const DeviceBuffer& curTailWeight = padRefine ? weightFp8Tail64 : weightsFp8[tailLayerIdx];
                    tailCudnn = std::make_unique<CudnnConvRuntime>();
                    tailCudnn->Configure(
                        {packedWidth,packedHeight,refineChannels,tailChannels,1,0,"tail",
                         CudnnConvEpilogue::Fp16Bias,
                         halfHead ? CudnnConvDataType::Fp16 : CudnnConvDataType::Fp8E4m3,
                         precision==RuntimePrecision::Fp8&&direct},stream,
                        {{halfHead ? packedFp16.Get() : workAFp8.Get(),
                          halfHead ? weightsFp16[tailLayerIdx].Get() : curTailWeight.Get(),halfHead ? nullptr : scale(expandOutActIdx + 2),
                          halfHead ? nullptr : weightScales[tailLayerIdx].As<float>(),Fp8Bias(tailLayerIdx),
                          nullptr,nullptr,tailFp16.Get()}});
                }
                else if (precision == RuntimePrecision::Mixed)
                {
                    auto scale = [this](size_t index)
                    {
                        return activationScalesDevice.As<float>() + index;
                    };
                    auto reciprocal = [this](size_t index)
                    {
                        return outputReciprocalScalesDevice.As<float>() + index;
                    };
                    constexpr CudnnConvDataType half = CudnnConvDataType::Fp16;
                    constexpr CudnnConvEpilogue preluHalf = CudnnConvEpilogue::Fp16Prelu;

                    projectCudnn = std::make_unique<CudnnConvRuntime>();
                    projectCudnn->Configure(
                        {packedWidth,packedHeight,packedChannels,channels540,1,0,"project",
                         preluHalf,half},stream,
                        {{packedFp16.Get(),weightsFp16[0].Get(),nullptr,nullptr,
                          biases[0].As<float>(),prelus[0].As<float>(),nullptr,
                          skipFp16.Get()}});
                    if (!hasFusedNativeFp8Compress)
                    {
                        compressCudnn = std::make_unique<CudnnConvRuntime>();
                        compressCudnn->Configure(
                            {lowWidth,lowHeight,channels270In,64,1,0,"compress"},stream,
                            {{workAFp8.Get(),weightsFp8[1].Get(),scale(1),
                              weightScales[1].As<float>(),Fp8Bias(1),
                              prelus[1].As<float>(),reciprocal(2),lowAFp8.Get()}});
                    }
                    else
                    {
                        compressCudnn.reset();
                    }
                    if (!hasParallel32x4Bottlenecks)
                    {
                        bottleneckCudnn = std::make_unique<CudnnConvRuntime>();
                        bottleneckCudnn->Configure(
                            {lowWidth,lowHeight,64,64,3,1,"bottleneck"},stream,
                            {
                                {lowAFp8.Get(),weightsFp8[2].Get(),scale(2),
                                 weightScales[2].As<float>(),Fp8Bias(2),
                                 prelus[2].As<float>(),reciprocal(3),lowBFp8.Get()},
                                {lowBFp8.Get(),weightsFp8[3].Get(),scale(3),
                                 weightScales[3].As<float>(),Fp8Bias(3),
                                 prelus[3].As<float>(),reciprocal(4),lowAFp8.Get()},
                            });
                    }
                    else
                    {
                        bottleneckCudnn.reset();
                    }
                    const size_t expandLayerIdx = hasParallel32x4Bottlenecks ? 6 : 4;
                    const size_t expandInActIdx = hasParallel32x4Bottlenecks ? 6 : 4;
                    const size_t expandOutActIdx = hasParallel32x4Bottlenecks ? 7 : 5;
                    if (!hasFusedNativeFp8Expand)
                    {
                        expandCudnn = std::make_unique<CudnnConvRuntime>();
                        expandCudnn->Configure(
                            {lowWidth,lowHeight,64,channels270Out,1,0,"expand"},stream,
                            {{lowAFp8.Get(),weightsFp8[expandLayerIdx].Get(),scale(expandInActIdx),
                              weightScales[expandLayerIdx].As<float>(),Fp8Bias(expandLayerIdx),
                              prelus[expandLayerIdx].As<float>(),reciprocal(expandOutActIdx),workAFp8.Get()}});
                    }
                    else
                    {
                        expandCudnn.reset();
                    }
                    const size_t refineLayerIdx = model.Layers().size() - 2;
                    if (!model.AdaptivePairMoE() && !hasFastWmmaRefine)
                    {
                        refineCudnn = std::make_unique<CudnnConvRuntime>();
                        refineCudnn->Configure(
                            {packedWidth,packedHeight,channels540,refineChannels,3,1,"refine",
                             preluHalf,half},stream,
                            {{workBFp16.Get(),weightsFp16[refineLayerIdx].Get(),nullptr,nullptr,
                              biases[refineLayerIdx].As<float>(),prelus[refineLayerIdx].As<float>(),nullptr,
                              workAFp16.Get()}});
                        tailCudnn.reset();
                        ConfigureTailCublasLt(packedWidth, packedHeight);
                    }
                    else
                    {
                        refineCudnn.reset();
                        tailCudnn.reset();
                        ConfigureTailCublasLt(packedWidth, packedHeight);
                    }
                }
                else
                {
                    constexpr CudnnConvDataType half=
                        CudnnConvDataType::Fp16;
                    constexpr CudnnConvEpilogue prelu=
                        CudnnConvEpilogue::Fp16Prelu;
                    projectCudnn = std::make_unique<CudnnConvRuntime>();
                    projectCudnn->Configure(
                        {packedWidth,packedHeight,packedChannels,channels540,1,0,"project",
                         prelu,half},stream,
                        {{packedFp16.Get(),weightsFp16[0].Get(),nullptr,nullptr,
                          biases[0].As<float>(),prelus[0].As<float>(),nullptr,
                          skipFp16.Get()}});
                    compressCudnn = std::make_unique<CudnnConvRuntime>();
                    compressCudnn->Configure(
                        {lowWidth,lowHeight,channels270In,64,1,0,"compress",
                         prelu,half},stream,
                        {{workAFp16.Get(),weightsFp16[1].Get(),nullptr,nullptr,
                          biases[1].As<float>(),prelus[1].As<float>(),nullptr,
                          lowAFp16.Get()}});
                    if (!hasParallel32x4Bottlenecks)
                    {
                        bottleneckCudnn = std::make_unique<CudnnConvRuntime>();
                        bottleneckCudnn->Configure(
                            {lowWidth,lowHeight,64,64,3,1,"bottleneck",
                             prelu,half},stream,
                            {
                                {lowAFp16.Get(),weightsFp16[2].Get(),nullptr,nullptr,
                                 biases[2].As<float>(),prelus[2].As<float>(),nullptr,
                                 lowBFp16.Get()},
                                {lowBFp16.Get(),weightsFp16[3].Get(),nullptr,nullptr,
                                 biases[3].As<float>(),prelus[3].As<float>(),nullptr,
                                 lowAFp16.Get()},
                            });
                    }
                    else
                    {
                        bottleneckCudnn.reset();
                    }
                    const size_t expandLayerIdxFp16 = hasParallel32x4Bottlenecks ? 6 : 4;
                    expandCudnn = std::make_unique<CudnnConvRuntime>();
                    expandCudnn->Configure(
                        {lowWidth,lowHeight,64,channels270Out,1,0,"expand",
                         prelu,half},stream,
                        {{lowAFp16.Get(),weightsFp16[expandLayerIdxFp16].Get(),nullptr,nullptr,
                          biases[expandLayerIdxFp16].As<float>(),prelus[expandLayerIdxFp16].As<float>(),nullptr,
                          workAFp16.Get()}});
                    const size_t refineLayerIdxFp16 = model.Layers().size() - 2;
                    if (!model.AdaptivePairMoE() && !hasFastWmmaRefine)
                    {
                        refineCudnn = std::make_unique<CudnnConvRuntime>();
                        refineCudnn->Configure(
                            {packedWidth,packedHeight,channels540,refineChannels,3,1,"refine",
                             prelu,half},stream,
                            {{workBFp16.Get(),weightsFp16[refineLayerIdxFp16].Get(),nullptr,nullptr,
                              biases[refineLayerIdxFp16].As<float>(),prelus[refineLayerIdxFp16].As<float>(),nullptr,
                              workAFp16.Get()}});
                        tailCudnn.reset();
                        ConfigureTailCublasLt(packedWidth, packedHeight);
                    }
                    else
                    {
                        refineCudnn.reset();
                        tailCudnn.reset();
                        ConfigureTailCublasLt(packedWidth, packedHeight);
                    }
                }

                persistentBytes = edge.Bytes() + accumulator.Bytes() +
                    tailFp16.Bytes() +
                    workspace.Bytes() + packedFp8.Bytes() + packedFp16.Bytes() +
                    activationScalesDevice.Bytes() +
                    outputReciprocalScalesDevice.Bytes() +
                    skipFp8.Bytes() + skipFp16.Bytes() +
                    workAFp8.Bytes() + workAFp16.Bytes() +
                    workBFp8.Bytes() + workBFp16.Bytes() +
                    lowAFp8.Bytes() + lowAFp16.Bytes() +
                    lowBFp8.Bytes() + lowBFp16.Bytes();
                for (size_t layer = 0; layer < model.Layers().size(); ++layer)
                    persistentBytes += weightsFp8[layer].Bytes() +
                        weightsFp16[layer].Bytes() + weightScales[layer].Bytes() +
                        biases[layer].Bytes() + biasesFp8[layer].Bytes() + prelus[layer].Bytes();
                if (hasFastWmmaRefine)
                    persistentBytes += refineWeightsPacked.Bytes();
                if (hasNvFp4Bottlenecks)
                    persistentBytes += weightsNvFp4Mma[0].Bytes() + weightsNvFp4Mma[1].Bytes() +
                        weightsNvFp4Scale[0].Bytes() + weightsNvFp4Scale[1].Bytes();
                if (hasParallel32x4Bottlenecks)
                {
                    for (size_t bIdx = 0; bIdx < 4; ++bIdx)
                        persistentBytes += weightsFp8Mma[bIdx].Bytes() + weightsFp16Mma[bIdx].Bytes();
                }
                else if (hasNativeFp8Bottlenecks)
                    persistentBytes += weightsFp8Mma[0].Bytes() + weightsFp8Mma[1].Bytes();
                if (hasFusedNativeFp8Expand)
                    persistentBytes += weightsFp8ExpandMma.Bytes();
                if (hasFusedNativeFp8Compress)
                    persistentBytes += weightsFp8CompressMma.Bytes();
                if (hasFastWmmaRefine)
                    persistentBytes += refineWeightsPacked.Bytes();
                if (hasFusedRefineTail)
                    persistentBytes += tailWeightsPacked.Bytes();
                if (projectCudnn)
                    persistentBytes += projectCudnn->WorkspaceBytes();
                if (compressCudnn)
                    persistentBytes += compressCudnn->WorkspaceBytes();
                if (bottleneckCudnn)
                    persistentBytes += bottleneckCudnn->WorkspaceBytes();
                if (expandCudnn)
                    persistentBytes += expandCudnn->WorkspaceBytes();
                if (refineCudnn)
                    persistentBytes += refineCudnn->WorkspaceBytes();
                if (tailCudnn)
                    persistentBytes += tailCudnn->WorkspaceBytes();
                if (hasTailCublasLt)
                    persistentBytes += cublasWorkspace.Bytes() + tailBiasFp16.Bytes();

                cudaDeviceProp properties{};
                CheckCuda(cudaGetDeviceProperties(&properties,cudaDevice),
                    "cudaGetDeviceProperties(SourcePyramid)");
                int runtimeVersion = 0;
                CheckCuda(cudaRuntimeGetVersion(&runtimeVersion),
                    "cudaRuntimeGetVersion(SourcePyramid)");
                std::ostringstream report;
                report << "SourcePyramid model=" << model.Path().string()
                    << " architecture=" << model.ArchitectureId()
                    << " artifact_version=" << kArtifactFormatVersion
                    << " sha256=" << model.ChecksumHex()
                    << " GPU=\"" << properties.name << "\" cc="
                    << properties.major << '.' << properties.minor
                    << " CUDA=" << runtimeVersion
                    << " cuDNN="
                    << (projectCudnn ? projectCudnn->LibraryVersion() : 0)
                    << " precision="
                    << (precision == RuntimePrecision::Fp8 ? "FP8-E4M3FN" : (precision == RuntimePrecision::Mixed ? "FP8/FP16-Mixed" : "FP16"))
                    << (hasNvFp4Bottlenecks ? " body=Blackwell-SM120a-FP4-TensorCore+cudnn-frontend" :
                        (hasParallel32x4Bottlenecks ? " body=Blackwell-SM120a-Parallel32x4-FP8-TensorCore" :
                        (hasNativeFp8Bottlenecks ? " body=Blackwell-SM120a-Native-FP8-TensorCore+cudnn-frontend" : " body=cudnn-frontend-TensorCore-FP32-accum")))
                    << " fp8_half_project=" << halfProject << " fp8_half_head=" << halfHead
                    << " fp8_internal="
                    << (precision == RuntimePrecision::Fp8 || precision == RuntimePrecision::Mixed
                        ? (hasNvFp4Bottlenecks ? "cudnn-frontend+blackwell-sm120a-native-fp4-tensorcore" :
                           (hasParallel32x4Bottlenecks ? "blackwell-sm120a-parallel32x4-fp8-tensorcore" :
                           (hasNativeFp8Bottlenecks ? "cudnn-frontend+blackwell-sm120a-native-fp8-tensorcore" : "cudnn-frontend-fused-conv-scale-bias-prelu-requant")))
                        : "disabled")
                    << " fp16_internal="
                    << (precision == RuntimePrecision::Fp16 || precision == RuntimePrecision::Mixed
                        ? "cudnn-frontend-fused-conv-bias-prelu"
                        : "disabled")
                    << (direct ? " tail=metadata64+fused-adaptive16x5-four-phases"
                               : " tail=cudnn-frontend-fused-conv-bias+cuda-pixelshuffle4-edgephase-compose")
                    << " source=" << width << 'x' << height
                    << " padded=" << paddedWidth << 'x' << paddedHeight
                    << " packed_channels=" << packedChannels << " packed=" << packedWidth << 'x' << packedHeight
                    << " source_pad=" << sourcePadX << ',' << sourcePadY
                    << " preprocessing=" << (hasFusedProject ? "fused-pack-project" : (direct ? "fused-rgby-pixelunshuffle2" : EasuKernelModeName(easuKernelMode)))
                    << " compress=pixel-unshuffle2+conv1x1"
                    << " refine=" << ((hasFastWmmaRefine && precision != RuntimePrecision::Fp8)
                        ? "native-sm120" : "cudnn")
                    << " scratch=precision-exclusive-half-activations+no-fp32-accumulator"
                    << (model.AdaptiveOc25() ? " base=fused-learned5x5 coefficient_transfer=oc25 basis=16x4x25-zero-dc edge=none" :
                        (direct ? " base=fused-bilinear basis=16x4x25-zero-dc edge=none" : " base=direct-output-surface edge=FP16"))
                    << " edge_bytes=" << edge.Bytes()
                    << " persistent_bytes=" << persistentBytes
                    << " execution_engine=shared-single-highest-priority-stream"
                    << " submission=cuda-graph interop=d3d11-fence-external-semaphore"
                    << " cuda_stream_priority=" << streamPriority;
                startupReport = report.str();
                configuredWidth = width;
                configuredHeight = height;
                configuredPrecision = precision;
                configurationValid = true;
                configuredSlotCount = 1;
            }
            catch (...)
            {
                configurationValid = false;
                configuredSlotCount = 0;
                configuredWidth = configuredHeight = 0;
                startupReport.clear();
                persistentBytes = 0;
                projectCudnn.reset();
                compressCudnn.reset();
                bottleneckCudnn.reset();
                expandCudnn.reset();
                refineCudnn.reset();
                tailCudnn.reset();
                DestroyCublasLt();
                throw;
            }
        }

        void DetachConfiguration() noexcept
        {
            if (configuredSlotCount != 0) --configuredSlotCount;
            if (configuredSlotCount == 0) DestroyCublasLt();
        }

        float* Fp8Bias(size_t layer) const noexcept
        {
            return model.AdaptiveFilter() ? biasesFp8[layer].As<float>() : biases[layer].As<float>();
        }

        uint64_t CommitSubmission() noexcept
        {
            lastSubmissionOrdinal = ++nextSubmissionOrdinal;
            return lastSubmissionOrdinal;
        }

        ModelArtifact model;
        int cudaDevice = -1;
        int streamPriority = 0;
        cudaStream_t stream = nullptr;
        IUnknown* d3dDeviceIdentity = nullptr;
        ID3D11DeviceContext4* d3dContext4 = nullptr;
        uint64_t nextSubmissionOrdinal = 0;
        uint64_t lastSubmissionOrdinal = 0;
        uint32_t configuredSlotCount = 0;
        uint32_t configuredWidth = 0;
        uint32_t configuredHeight = 0;
        uint32_t sourcePadX = 0;
        uint32_t sourcePadY = 0;
        uint32_t paddedWidth = 0;
        uint32_t paddedHeight = 0;
        uint32_t packedWidth = 0;
        uint32_t packedHeight = 0;
        uint32_t projectInputChannels = 16;
        uint32_t refineChannels = 48;
        uint32_t lowWidth = 0;
        uint32_t lowHeight = 0;
        uint32_t outputWidth = 0;
        uint32_t outputHeight = 0;
        RuntimePrecision configuredPrecision = RuntimePrecision::Fp8;
        EasuKernelMode easuKernelMode = EasuKernelMode::TiledFp16;
        bool configurationValid = false;
        uint64_t persistentBytes = 0;
        std::string startupReport;
        std::vector<float> activationScales;
        DeviceBuffer activationScalesDevice;
        DeviceBuffer outputReciprocalScalesDevice;
        static constexpr size_t kMaxLayers = 9;
        std::array<DeviceBuffer,kMaxLayers> weightsFp8;
        std::array<DeviceBuffer,kMaxLayers> weightsFp16;
        std::array<DeviceBuffer,kMaxLayers> weightScales;
        std::array<DeviceBuffer,kMaxLayers> biases;
        std::array<DeviceBuffer,kMaxLayers> biasesFp8;
        std::array<DeviceBuffer,kMaxLayers> prelus;
        bool hasNvFp4Bottlenecks = false;
        std::array<DeviceBuffer,2> weightsNvFp4Mma;
        std::array<DeviceBuffer,2> weightsNvFp4Scale;
        bool hasParallel32x4Bottlenecks = false;
        bool hasNativeFp8Bottlenecks = false;
        std::array<DeviceBuffer,4> weightsFp8Mma;
        std::array<DeviceBuffer,4> weightsFp16Mma;
        bool hasFusedNativeFp8Expand = false;
        DeviceBuffer weightsFp8ExpandMma;
        bool hasFusedNativeFp8Compress = false;
        DeviceBuffer weightsFp8CompressMma;
        DeviceBuffer weightFp8Refine64;
        DeviceBuffer weightScaleRefine64;
        DeviceBuffer biasRefine64;
        DeviceBuffer biasFp8Refine64;
        DeviceBuffer preluRefine64;
        DeviceBuffer weightFp8Tail64;
        std::array<PlanInfo,kMaxLayers> plans{};
        aess::adaptive::BasisBank adaptiveBasisBank{};
        aess::adaptive::LearnedFilterBank adaptiveLearnedBank{};
        aess::adaptive::LearnedFilterBankHalf adaptiveLearnedBankHalf{};
        aess::adaptive::LearnedPairMoEFilterBank adaptiveLearnedMoEBank{};
        DeviceBuffer routeThresholdDevice;
        DeviceBuffer routeMetaFp16;
        DeviceBuffer edge;
        DeviceBuffer accumulator;
        DeviceBuffer tailFp16;
        DeviceBuffer workspace;
        DeviceBuffer packedFp8, packedFp16;
        DeviceBuffer skipFp8, skipFp16;
        DeviceBuffer workAFp8, workAFp16;
        DeviceBuffer workBFp8, workBFp16;
        DeviceBuffer lowAFp8, lowAFp16;
        DeviceBuffer lowBFp8, lowBFp16;
        std::unique_ptr<CudnnConvRuntime> projectCudnn;
        std::unique_ptr<CudnnConvRuntime> compressCudnn;
        std::unique_ptr<CudnnConvRuntime> bottleneckCudnn;
        std::unique_ptr<CudnnConvRuntime> expandCudnn;
        std::unique_ptr<CudnnConvRuntime> refineCudnn;
        std::unique_ptr<CudnnConvRuntime> tailCudnn;
        cublasLtHandle_t cublasLt = nullptr;
        cublasLtMatmulDesc_t tailMatmulDesc = nullptr;
        cublasLtMatrixLayout_t tailAdesc = nullptr;
        cublasLtMatrixLayout_t tailBdesc = nullptr;
        cublasLtMatrixLayout_t tailCdesc = nullptr;
        cublasLtMatrixLayout_t tailDdesc = nullptr;
        cublasLtMatmulPreference_t tailPref = nullptr;
        cublasLtMatmulHeuristicResult_t tailHeuristicResult{};
        bool hasTailCublasLt = false;
        DeviceBuffer tailBiasFp16;
        DeviceBuffer cublasWorkspace;
        bool hasFastWmmaRefine = false;
        bool useNativeFp16Refine = UseNativeFp16Refine();
        DeviceBuffer refineWeightsPacked;
        bool hasFusedRefineTail = false;
        bool useFusedTail = UseFusedTail();
        DeviceBuffer tailWeightsPacked;
        bool hasFusedProject = false;
        bool useFusedProject = UseFusedProject();
    };

    struct CudaRuntime::Impl
    {
        struct GraphVariant
        {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            cudaGraphNode_t waitNode = nullptr;
            cudaGraphNode_t signalNode = nullptr;

            ~GraphVariant() { Reset(); }
            GraphVariant() = default;
            GraphVariant(const GraphVariant&) = delete;
            GraphVariant& operator=(const GraphVariant&) = delete;
            GraphVariant(GraphVariant&& other) noexcept
                : graph(std::exchange(other.graph,nullptr)),
                  executable(std::exchange(other.executable,nullptr)),
                  waitNode(std::exchange(other.waitNode,nullptr)),
                  signalNode(std::exchange(other.signalNode,nullptr))
            {
            }
            GraphVariant& operator=(GraphVariant&& other) noexcept
            {
                if (this != &other)
                {
                    Reset();
                    graph = std::exchange(other.graph,nullptr);
                    executable = std::exchange(other.executable,nullptr);
                    waitNode = std::exchange(other.waitNode,nullptr);
                    signalNode = std::exchange(other.signalNode,nullptr);
                }
                return *this;
            }
            void Reset() noexcept
            {
                if (executable != nullptr) cudaGraphExecDestroy(executable);
                if (graph != nullptr) cudaGraphDestroy(graph);
                graph = nullptr;
                executable = nullptr;
                waitNode = nullptr;
                signalNode = nullptr;
            }
        };

        explicit Impl(const std::filesystem::path& path)
            : engine(std::make_shared<SharedExecutionEngine>(path))
        {
            CreateCompletionWakeEvent();
        }

        explicit Impl(std::shared_ptr<SharedExecutionEngine> shared)
            : engine(std::move(shared))
        {
            if (!engine)
                throw std::invalid_argument(
                    "SourcePyramid sibling requires a shared execution engine");
            CreateCompletionWakeEvent();
        }

        ~Impl()
        {
            if (engine && engine->cudaDevice >= 0) cudaSetDevice(engine->cudaDevice);
            ResetInterop();
            if (ownershipTimingEvent != nullptr)
                cudaEventDestroy(ownershipTimingEvent);
            if (completionEvent != nullptr) cudaEventDestroy(completionEvent);
            for (cudaEvent_t event : events)
                if (event != nullptr) cudaEventDestroy(event);
            if (completionWakeEvent != nullptr) CloseHandle(completionWakeEvent);
        }

        void CreateCompletionWakeEvent()
        {
            completionWakeEvent = CreateEventW(nullptr,FALSE,FALSE,nullptr);
            if (completionWakeEvent == nullptr)
                throw std::runtime_error(
                    "SourcePyramid could not create its completion wake event");
        }

        void EnsureSlotEvents()
        {
            if (completionEvent != nullptr) return;
            std::array<cudaEvent_t,18> newEvents{};
            cudaEvent_t newOwnership = nullptr;
            cudaEvent_t newCompletion = nullptr;
            try
            {
                for (cudaEvent_t& event : newEvents)
                    CheckCuda(cudaEventCreate(&event),
                        "cudaEventCreate(SourcePyramid timing)");
                CheckCuda(cudaEventCreate(&newOwnership),
                    "cudaEventCreate(SourcePyramid ownership)");
                CheckCuda(cudaEventCreateWithFlags(
                    &newCompletion,cudaEventBlockingSync|cudaEventDisableTiming),
                    "cudaEventCreateWithFlags(SourcePyramid completion)");
            }
            catch (...)
            {
                if (newCompletion != nullptr) cudaEventDestroy(newCompletion);
                if (newOwnership != nullptr) cudaEventDestroy(newOwnership);
                for (cudaEvent_t event : newEvents)
                    if (event != nullptr) cudaEventDestroy(event);
                throw;
            }
            events = newEvents;
            ownershipTimingEvent = newOwnership;
            completionEvent = newCompletion;
        }

        void CreateSlotSynchronization(ID3D11Device* device)
        {
            ID3D11Device5* device5 = nullptr;
            HRESULT hr = device->QueryInterface(
                __uuidof(ID3D11Device5),reinterpret_cast<void**>(&device5));
            if (FAILED(hr) || device5 == nullptr)
                throw std::runtime_error("SourcePyramid requires ID3D11Device5");
            ID3D11Fence* newFence = nullptr;
            hr = device5->CreateFence(
                0,D3D11_FENCE_FLAG_SHARED,__uuidof(ID3D11Fence),
                reinterpret_cast<void**>(&newFence));
            device5->Release();
            if (FAILED(hr) || newFence == nullptr)
                throw std::runtime_error("SourcePyramid could not create its D3D11 fence");
            HANDLE handle = nullptr;
            hr = newFence->CreateSharedHandle(nullptr,GENERIC_ALL,nullptr,&handle);
            if (FAILED(hr) || handle == nullptr)
            {
                newFence->Release();
                throw std::runtime_error("SourcePyramid could not share its D3D11 fence");
            }
            cudaExternalSemaphoreHandleDesc descriptor{};
            descriptor.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
            descriptor.handle.win32.handle = handle;
            cudaExternalSemaphore_t semaphore = nullptr;
            const cudaError_t status = cudaImportExternalSemaphore(
                &semaphore,&descriptor);
            CloseHandle(handle);
            if (status != cudaSuccess)
            {
                newFence->Release();
                CheckCuda(status,"cudaImportExternalSemaphore(SourcePyramid)");
            }
            d3dFence = newFence;
            externalSemaphore = semaphore;
            synchronizationValue = 0;
        }

        static size_t GraphIndex(
            bool detailed,
            RuntimeOutputMode outputMode) noexcept
        {
            return (outputMode == RuntimeOutputMode::EasuOnly ? 2u : 0u) +
                (detailed ? 1u : 0u);
        }

        void DestroyGraphs() noexcept
        {
            for (GraphVariant& graph : graphs) graph.Reset();
        }

        void Record(cudaEvent_t event, const char* operation)
        {
            CheckCuda(eagerValidation ? cudaEventRecord(event,engine->stream) :
                cudaEventRecordWithFlags(event,engine->stream,cudaEventRecordExternal),operation);
            if (eagerValidation) CheckCuda(cudaStreamSynchronize(engine->stream),operation);
        }

        void RunConvolution(size_t layer)
        {
            if (configuredPrecision == RuntimePrecision::Fp8)
                CheckCutlass(engine->plans[layer].fp8(engine->stream),
                    "SourcePyramid CUTLASS FP8 execute");
            else
                CheckCutlass(engine->plans[layer].fp16(engine->stream),
                    "SourcePyramid CUTLASS FP16 execute");
        }

        void RunEpilogue(size_t layer, void* output)
        {
            const size_t count = engine->plans[layer].outputElements;
            constexpr int threads = 256;
            const int blocks = int((count + threads - 1) / threads);
            if (configuredPrecision == RuntimePrecision::Fp8)
            {
                constexpr int inputScaleIndex[6] = {0,1,2,3,4,6};
                constexpr int outputScaleIndex[6] = {1,2,3,4,5,7};
                ApplyFp8EpilogueKernel<<<blocks,threads,0,engine->stream>>>(
                    engine->accumulator.As<float>(),
                    reinterpret_cast<__nv_fp8_e4m3*>(output),count,
                    engine->plans[layer].outputChannels,
                    engine->activationScales[inputScaleIndex[layer]],
                    engine->activationScales[outputScaleIndex[layer]],
                    engine->weightScales[layer].As<float>(),
                    engine->biases[layer].As<float>(),
                    engine->prelus[layer].As<float>());
            }
            else
            {
                ApplyHalfEpilogueKernel<<<blocks,threads,0,engine->stream>>>(
                    engine->accumulator.As<float>(),
                    reinterpret_cast<Half*>(output),count,
                    engine->plans[layer].outputChannels,
                    engine->biases[layer].As<float>(),
                    engine->prelus[layer].As<float>());
            }
            CheckLaunch("SourcePyramid epilogue");
        }

        void EnqueueGraphBody(
            bool detailed,
            RuntimeOutputMode outputMode,
            uint64_t inputReadyValue,
            uint64_t outputReadyValue)
        {
            cudaExternalSemaphoreWaitParams wait{};
            wait.params.fence.value = inputReadyValue;
            CheckCuda(cudaWaitExternalSemaphoresAsync(
                &externalSemaphore,&wait,1,engine->stream),
                "cudaWaitExternalSemaphoresAsync(SourcePyramid capture)");
            Record(events[0],"record SourcePyramid frame begin");

            const dim3 imageBlock(16,16);
            const dim3 imageGrid(
                (engine->paddedWidth*2+imageBlock.x-1)/imageBlock.x,
                (engine->paddedHeight*2+imageBlock.y-1)/imageBlock.y);
            const __half edgeGateHalf=__float2half_rn(
                std::max(engine->model.EdgeGateGain(),0.0f));
            if (engine->model.AdaptiveFilter())
            {
                if (outputMode != RuntimeOutputMode::Neural)
                    throw std::logic_error("Adaptive filter has no EASU base; use the EASU model's debug graph");
                const int blocks = int((size_t(engine->packedWidth) * engine->packedHeight + 255) / 256);
                if (configuredPrecision==RuntimePrecision::Fp16 || configuredPrecision==RuntimePrecision::Mixed || engine->model.Fp8HalfProject())
                {
                    if (engine->model.AdaptivePairMoE())
                    {
                        PackLumaKernel<Half><<<blocks,256,0,engine->stream>>>(externalInput.Surface(),
                            int(inputWidth),int(inputHeight),int(engine->packedWidth),int(engine->packedHeight),
                            int(engine->sourcePadX),int(engine->sourcePadY),engine->packedFp16.As<Half>(),1.0f);
                    }
                    else if (engine->model.AdaptiveLearnedRgb4Dir())
                    {
                        const dim3 block2D(16, 16);
                        const dim3 grid2D((engine->packedWidth + 15) / 16, (engine->packedHeight + 15) / 16);
                        if (engine->hasFusedProject && (configuredPrecision == RuntimePrecision::Mixed || configuredPrecision == RuntimePrecision::Fp16))
                        {
                            FusedPackRgbYLearnedRgb4DirAndProjectKernel<<<grid2D, block2D, 0, engine->stream>>>(
                                externalInput.Surface(),
                                int(inputWidth), int(inputHeight),
                                int(engine->packedWidth), int(engine->packedHeight),
                                int(engine->sourcePadX), int(engine->sourcePadY),
                                engine->skipFp16.As<half>(),
                                engine->weightsFp16[0].As<half>(),
                                engine->biases[0].As<float>(),
                                engine->prelus[0].As<float>(),
                                nullptr,
                                engine->model.Analysis4Dir());
                        }
                        else
                        {
                            PackRgbYLearnedRgb4DirTiledKernel<Half><<<grid2D, block2D, 0, engine->stream>>>(externalInput.Surface(),
                                int(inputWidth), int(inputHeight), int(engine->packedWidth), int(engine->packedHeight),
                                int(engine->sourcePadX), int(engine->sourcePadY), engine->packedFp16.As<Half>(), 1.0f,
                                engine->model.Analysis4Dir());
                        }
                    }
                    else if (engine->model.AdaptiveFeature8())
                    {
                        const dim3 block2D(16, 16);
                        const dim3 grid2D((engine->packedWidth + 15) / 16, (engine->packedHeight + 15) / 16);
                        PackRgbYFeature8TiledKernel<Half><<<grid2D, block2D, 0, engine->stream>>>(externalInput.Surface(),
                            int(inputWidth), int(inputHeight), int(engine->packedWidth), int(engine->packedHeight),
                            int(engine->sourcePadX), int(engine->sourcePadY), engine->packedFp16.As<Half>(), 1.0f,
                            engine->model.AnalysisStencil());
                    }
                    else
                        PackRgbYKernel<Half,16,false><<<blocks,256,0,engine->stream>>>(externalInput.Surface(),
                            int(inputWidth),int(inputHeight),int(engine->packedWidth),int(engine->packedHeight),
                            int(engine->sourcePadX),int(engine->sourcePadY),engine->packedFp16.As<Half>(),1.0f);
                }
                else if (engine->model.AdaptiveLearnedRgb4Dir())
                {
                    const dim3 block2D(16, 16);
                    const dim3 grid2D((engine->packedWidth + 15) / 16, (engine->packedHeight + 15) / 16);
                    PackRgbYLearnedRgb4DirTiledKernel<__nv_fp8_e4m3><<<grid2D, block2D, 0, engine->stream>>>(externalInput.Surface(),
                        int(inputWidth), int(inputHeight), int(engine->packedWidth), int(engine->packedHeight),
                        int(engine->sourcePadX), int(engine->sourcePadY), engine->packedFp8.As<__nv_fp8_e4m3>(),
                        1.0f / engine->activationScales[0],
                        engine->model.Analysis4Dir());
                }
                else if (engine->model.AdaptiveFeature8())
                {
                    const dim3 block2D(16, 16);
                    const dim3 grid2D((engine->packedWidth + 15) / 16, (engine->packedHeight + 15) / 16);
                    PackRgbYFeature8TiledKernel<__nv_fp8_e4m3><<<grid2D, block2D, 0, engine->stream>>>(externalInput.Surface(),
                        int(inputWidth), int(inputHeight), int(engine->packedWidth), int(engine->packedHeight),
                        int(engine->sourcePadX), int(engine->sourcePadY), engine->packedFp8.As<__nv_fp8_e4m3>(),
                        1.0f / engine->activationScales[0],
                        engine->model.AnalysisStencil());
                }
                else if (engine->projectInputChannels == 32)
                    PackRgbYKernel<__nv_fp8_e4m3,32,false><<<blocks,256,0,engine->stream>>>(externalInput.Surface(),
                        int(inputWidth),int(inputHeight),int(engine->packedWidth),int(engine->packedHeight),
                        int(engine->sourcePadX),int(engine->sourcePadY),engine->packedFp8.As<__nv_fp8_e4m3>(),
                        1.0f/engine->activationScales[0]);
                else
                    PackRgbYKernel<__nv_fp8_e4m3,16,false><<<blocks,256,0,engine->stream>>>(externalInput.Surface(),
                        int(inputWidth),int(inputHeight),int(engine->packedWidth),int(engine->packedHeight),
                        int(engine->sourcePadX),int(engine->sourcePadY),engine->packedFp8.As<__nv_fp8_e4m3>(),
                        1.0f/engine->activationScales[0]);
            }
            else if (configuredPrecision == RuntimePrecision::Fp8)
            {
                if (engine->easuKernelMode == EasuKernelMode::TiledFp16)
                {
                    EasuPrepareTiledHalfKernel<16,16,__nv_fp8_e4m3><<<
                        imageGrid,imageBlock,0,engine->stream>>>(
                        externalInput.Surface(),int(inputWidth),int(inputHeight),
                        int(engine->paddedWidth),int(engine->paddedHeight),8,8,
                        int(engine->outputWidth),int(engine->outputHeight),
                        engine->edge.As<ushort4>(),
                        engine->packedFp8.As<__nv_fp8_e4m3>(),
                        1.0f/engine->activationScales[0],
                        edgeGateHalf,
                        externalOutput.Surface());
                }
                else
                {
                    EasuPrepareReferenceKernel<__nv_fp8_e4m3><<<
                        imageGrid,imageBlock,0,engine->stream>>>(
                        externalInput.Surface(),int(inputWidth),int(inputHeight),
                        int(engine->paddedWidth),int(engine->paddedHeight),8,8,
                        int(engine->outputWidth),int(engine->outputHeight),
                        engine->edge.As<ushort4>(),
                        engine->packedFp8.As<__nv_fp8_e4m3>(),
                        1.0f/engine->activationScales[0],
                        engine->model.EdgeGateGain(),
                        externalOutput.Surface());
                }
            }
            else
            {
                if (engine->easuKernelMode == EasuKernelMode::TiledFp16)
                {
                    EasuPrepareTiledHalfKernel<16,16,Half><<<
                        imageGrid,imageBlock,0,engine->stream>>>(
                        externalInput.Surface(),int(inputWidth),int(inputHeight),
                        int(engine->paddedWidth),int(engine->paddedHeight),8,8,
                        int(engine->outputWidth),int(engine->outputHeight),
                        engine->edge.As<ushort4>(),engine->packedFp16.As<Half>(),
                        1.0f,edgeGateHalf,externalOutput.Surface());
                }
                else
                {
                    EasuPrepareReferenceKernel<Half><<<
                        imageGrid,imageBlock,0,engine->stream>>>(
                        externalInput.Surface(),int(inputWidth),int(inputHeight),
                        int(engine->paddedWidth),int(engine->paddedHeight),8,8,
                        int(engine->outputWidth),int(engine->outputHeight),
                        engine->edge.As<ushort4>(),engine->packedFp16.As<Half>(),
                        1.0f,engine->model.EdgeGateGain(),externalOutput.Surface());
                }
            }
            CheckLaunch(engine->hasFusedProject ? "SourcePyramid Fused Pack & Project" : "SourcePyramid EASU/pack80");
            if (detailed) Record(events[1],"record SourcePyramid EASU end");
            if (outputMode == RuntimeOutputMode::EasuOnly)
            {
                // Both SourcePyramid preprocessing variants already publish
                // their exact EASU base directly into the output surface.
                Record(events[16],"record SourcePyramid EASU-only frame end");
                cudaExternalSemaphoreSignalParams signal{};
                signal.params.fence.value = outputReadyValue;
                CheckCuda(cudaSignalExternalSemaphoresAsync(
                    &externalSemaphore,&signal,1,engine->stream),
                    "cudaSignalExternalSemaphoresAsync(SourcePyramid EASU-only capture)");
                if (detailed) Record(ownershipTimingEvent,
                    "record SourcePyramid EASU-only ownership transfer");
                Record(completionEvent,
                    "record SourcePyramid EASU-only completion");
                return;
            }

            constexpr int threads=256;
            if (configuredPrecision == RuntimePrecision::Fp8)
            {
                engine->projectCudnn->Execute(0);
                if (detailed) Record(events[2],
                    "record SourcePyramid fused project conv end");
                const uint32_t c540 = engine->model.Channels540();
                const size_t unshuffleElements =
                    size_t(engine->lowHeight)*engine->lowWidth*c540;
                const int blocks=int((unshuffleElements+threads-1)/threads);
                if (c540 == 32)
                    PixelUnshuffle2Fp8Kernel<32><<<blocks,threads,0,engine->stream>>>(
                        engine->skipFp8.As<__nv_fp8_e4m3>(),
                        engine->workAFp8.As<__nv_fp8_e4m3>(),
                        int(engine->lowHeight),int(engine->lowWidth));
                else
                    PixelUnshuffle2Fp8Kernel<48><<<blocks,threads,0,engine->stream>>>(
                        engine->skipFp8.As<__nv_fp8_e4m3>(),
                        engine->workAFp8.As<__nv_fp8_e4m3>(),
                        int(engine->lowHeight),int(engine->lowWidth));
            }
            else if (configuredPrecision == RuntimePrecision::Mixed)
            {
                if (!engine->hasFusedProject)
                {
                    engine->projectCudnn->Execute(0);
                }
                if (detailed) Record(events[2],
                    "record SourcePyramid fused project conv end");
                if (!engine->hasFusedNativeFp8Compress)
                {
                    const uint32_t c540 = engine->model.Channels540();
                    const size_t unshuffleElements =
                        size_t(engine->lowHeight)*engine->lowWidth*c540;
                    const int blocks=int((unshuffleElements+threads-1)/threads);
                    if (c540 == 32)
                        PixelUnshuffle2QuantizeFp8Kernel<32><<<blocks,threads,0,engine->stream>>>(
                            engine->skipFp16.As<Half>(),
                            engine->workAFp8.As<__nv_fp8_e4m3>(),
                            int(engine->lowHeight),int(engine->lowWidth),
                            1.0f / engine->activationScales[1]);
                    else
                        PixelUnshuffle2QuantizeFp8Kernel<48><<<blocks,threads,0,engine->stream>>>(
                            engine->skipFp16.As<Half>(),
                            engine->workAFp8.As<__nv_fp8_e4m3>(),
                            int(engine->lowHeight),int(engine->lowWidth),
                            1.0f / engine->activationScales[1]);
                }
            }
            else
            {
                if (!engine->hasFusedProject)
                {
                    engine->projectCudnn->Execute(0);
                }
                if (detailed) Record(events[2],
                    "record SourcePyramid fused project conv end");
                const uint32_t c540 = engine->model.Channels540();
                const size_t unshuffleElements =
                    size_t(engine->lowHeight)*engine->lowWidth*c540;
                const int blocks=int((unshuffleElements+threads-1)/threads);
                if (c540 == 32)
                    PixelUnshuffle2HalfKernel<32><<<blocks,threads,0,engine->stream>>>(
                        engine->skipFp16.As<Half>(),engine->workAFp16.As<Half>(),
                        int(engine->lowHeight),int(engine->lowWidth));
                else
                    PixelUnshuffle2HalfKernel<48><<<blocks,threads,0,engine->stream>>>(
                        engine->skipFp16.As<Half>(),engine->workAFp16.As<Half>(),
                        int(engine->lowHeight),int(engine->lowWidth));
            }
            CheckLaunch("SourcePyramid fused project epilogue/pixel-unshuffle2");
            if (detailed)
            {
                Record(events[3],"record SourcePyramid project epilogue end");
                Record(events[4],"record SourcePyramid fused unshuffle end");
            }

            constexpr int convEvents[3]={5,7,9};
            constexpr int epiEvents[3]={6,8,10};
            if (configuredPrecision == RuntimePrecision::Fp8)
            {
                engine->compressCudnn->Execute(0);
                if (detailed)
                {
                    Record(events[5],"record SourcePyramid fused compress end");
                    Record(events[6],"record SourcePyramid compress epilogue end");
                }
            }
            else if (configuredPrecision == RuntimePrecision::Mixed)
            {
                if (engine->hasFusedNativeFp8Compress)
                {
                    LaunchFusedNativeFp8Compress(
                        engine->stream,
                        engine->lowWidth, engine->lowHeight,
                        engine->skipFp16.Get(),
                        engine->lowAFp8.Get(),
                        engine->weightsFp8CompressMma.Get(),
                        1.0f / engine->activationScales[1],
                        engine->activationScales[1],
                        engine->weightScales[1].As<float>(),
                        engine->Fp8Bias(1),
                        engine->prelus[1].As<float>(),
                        1.0f / engine->activationScales[2]);
                    if (detailed)
                    {
                        Record(events[5],"record SourcePyramid fused compress end");
                        Record(events[6],"record SourcePyramid compress epilogue end");
                    }
                }
                else
                {
                    engine->compressCudnn->Execute(0);
                    if (detailed)
                    {
                        Record(events[5],"record SourcePyramid fused compress end");
                        Record(events[6],"record SourcePyramid compress epilogue end");
                    }
                }
            }
            if (configuredPrecision == RuntimePrecision::Fp8 || configuredPrecision == RuntimePrecision::Mixed)
            {
                if (engine->hasParallel32x4Bottlenecks)
                {
                    for (size_t bIdx = 0; bIdx < 4; ++bIdx)
                    {
                        const void* inPtr = (bIdx % 2 == 0) ? engine->lowAFp8.Get() : engine->lowBFp8.Get();
                        void* outPtr = (bIdx % 2 == 0) ? engine->lowBFp8.Get() : engine->lowAFp8.Get();
                        size_t layerIdx = bIdx + 2;
                        float actScale = engine->activationScales[bIdx + 2];
                        float outRecip = 1.0f / engine->activationScales[bIdx + 3];
                        LaunchNativeFp8Grouped64Conv(
                            engine->stream,
                            engine->lowWidth, engine->lowHeight,
                            inPtr, outPtr,
                            engine->weightsFp8Mma[bIdx].Get(),
                            actScale,
                            engine->weightScales[layerIdx].As<float>(),
                            engine->Fp8Bias(layerIdx),
                            engine->prelus[layerIdx].As<float>(),
                            outRecip);
                        if (detailed && bIdx == 1)
                        {
                            Record(events[7], "record SourcePyramid fused bottleneck end");
                            Record(events[8], "record SourcePyramid bottleneck epilogue end");
                        }
                    }
                    if (detailed)
                    {
                        Record(events[9], "record SourcePyramid fused bottleneck end");
                        Record(events[10], "record SourcePyramid bottleneck epilogue end");
                    }
                }
                else
                {
                    for (size_t variant=0;variant<2;++variant)
                    {
                        if (engine->hasNvFp4Bottlenecks)
                        {
                            const void* inPtr = (variant == 0) ? engine->lowAFp8.Get() : engine->lowBFp8.Get();
                            void* outPtr = (variant == 0) ? engine->lowBFp8.Get() : engine->lowAFp8.Get();
                            size_t layerIdx = variant + 2;
                            float actScale = engine->activationScales[variant + 2];
                            float outRecip = 1.0f / engine->activationScales[variant + 3];
                            LaunchBlackwellFp4BottleneckConv(
                                engine->stream,
                                engine->lowWidth, engine->lowHeight,
                                inPtr, outPtr,
                                engine->weightsNvFp4Mma[variant].Get(),
                                actScale,
                                engine->weightsNvFp4Scale[variant].As<float>(),
                                engine->Fp8Bias(layerIdx),
                                engine->prelus[layerIdx].As<float>(),
                                outRecip);
                        }
                        else if (engine->hasNativeFp8Bottlenecks)
                        {
                            const void* inPtr = (variant == 0) ? engine->lowAFp8.Get() : engine->lowBFp8.Get();
                            void* outPtr = (variant == 0) ? engine->lowBFp8.Get() : engine->lowAFp8.Get();
                            size_t layerIdx = variant + 2;
                            float actScale = engine->activationScales[variant + 2];
                            float outRecip = 1.0f / engine->activationScales[variant + 3];
                            LaunchNativeFp8BottleneckConv(
                                engine->stream,
                                engine->lowWidth, engine->lowHeight,
                                inPtr, outPtr,
                                engine->weightsFp8Mma[variant].Get(),
                                actScale,
                                engine->weightScales[layerIdx].As<float>(),
                                engine->Fp8Bias(layerIdx),
                                engine->prelus[layerIdx].As<float>(),
                                outRecip);
                        }
                        else
                        {
                            engine->bottleneckCudnn->Execute(variant);
                        }
                        if (detailed)
                        {
                            Record(events[convEvents[variant+1]],
                                "record SourcePyramid fused bottleneck end");
                            Record(events[epiEvents[variant+1]],
                                "record SourcePyramid bottleneck epilogue end");
                        }
                    }
                }
            }
            else
            {
                engine->compressCudnn->Execute(0);
                if (detailed)
                {
                    Record(events[5],"record SourcePyramid fused compress end");
                    Record(events[6],"record SourcePyramid compress epilogue end");
                }
                if (engine->hasParallel32x4Bottlenecks)
                {
                    for (size_t bIdx = 0; bIdx < 4; ++bIdx)
                    {
                        const void* inPtr = (bIdx % 2 == 0) ? engine->lowAFp16.Get() : engine->lowBFp16.Get();
                        void* outPtr = (bIdx % 2 == 0) ? engine->lowBFp16.Get() : engine->lowAFp16.Get();
                        size_t layerIdx = bIdx + 2;
                        LaunchNativeFp16Grouped64Conv(
                            engine->stream,
                            engine->lowWidth, engine->lowHeight,
                            inPtr, outPtr,
                            engine->weightsFp16Mma[bIdx].Get(),
                            engine->biases[layerIdx].As<float>(),
                            engine->prelus[layerIdx].As<float>());
                        if (detailed && bIdx == 1)
                        {
                            Record(events[7], "record SourcePyramid fused bottleneck end");
                            Record(events[8], "record SourcePyramid bottleneck epilogue end");
                        }
                    }
                    if (detailed)
                    {
                        Record(events[9], "record SourcePyramid fused bottleneck end");
                        Record(events[10], "record SourcePyramid bottleneck epilogue end");
                    }
                }
                else
                {
                    for (size_t variant=0;variant<2;++variant)
                    {
                        engine->bottleneckCudnn->Execute(variant);
                        if (detailed)
                        {
                            Record(events[convEvents[variant+1]],
                                "record SourcePyramid fused bottleneck end");
                            Record(events[epiEvents[variant+1]],
                                "record SourcePyramid bottleneck epilogue end");
                        }
                    }
                }
            }

            const size_t expandLayerIdx = engine->hasParallel32x4Bottlenecks ? 6 : 4;
            const size_t expandInActIdx = engine->hasParallel32x4Bottlenecks ? 6 : 4;
            const size_t expandOutActIdx = engine->hasParallel32x4Bottlenecks ? 7 : 5;
            if (configuredPrecision == RuntimePrecision::Fp8)
            {
                engine->expandCudnn->Execute(0);
                if(detailed) Record(events[11],
                    "record SourcePyramid fused expand conv end");
                const uint32_t c540 = engine->model.Channels540();
                const size_t shuffleElements=
                    size_t(engine->lowHeight)*engine->lowWidth*c540;
                const int blocks=int((shuffleElements+threads-1)/threads);
                if (c540 == 32)
                    PixelShuffle2SkipAddFp8Kernel<32><<<blocks,threads,0,engine->stream>>>(
                        engine->workAFp8.As<__nv_fp8_e4m3>(),engine->skipFp8.As<__nv_fp8_e4m3>(),
                        engine->workBFp8.As<__nv_fp8_e4m3>(),int(engine->lowHeight),int(engine->lowWidth),
                        engine->activationScales[expandOutActIdx],engine->activationScales[1],1.0f/engine->activationScales[expandOutActIdx + 1]);
                else
                    PixelShuffle2SkipAddFp8Kernel<48><<<blocks,threads,0,engine->stream>>>(
                        engine->workAFp8.As<__nv_fp8_e4m3>(),engine->skipFp8.As<__nv_fp8_e4m3>(),
                        engine->workBFp8.As<__nv_fp8_e4m3>(),int(engine->lowHeight),int(engine->lowWidth),
                        engine->activationScales[expandOutActIdx],engine->activationScales[1],1.0f/engine->activationScales[expandOutActIdx + 1]);
            }
            else if (configuredPrecision == RuntimePrecision::Mixed)
            {
                if (engine->hasFusedNativeFp8Expand)
                {
                    LaunchFusedNativeFp8Expand(
                        engine->stream,
                        engine->lowWidth, engine->lowHeight,
                        engine->lowAFp8.Get(),
                        engine->skipFp16.Get(),
                        engine->workBFp16.Get(),
                        engine->weightsFp8ExpandMma.Get(),
                        engine->activationScales[expandInActIdx],
                        engine->weightScales[expandLayerIdx].As<float>(),
                        engine->Fp8Bias(expandLayerIdx),
                        engine->prelus[expandLayerIdx].As<float>(),
                        1.0f / engine->activationScales[expandOutActIdx],
                        engine->activationScales[expandOutActIdx]);
                    if (detailed) Record(events[11],
                        "record SourcePyramid fused expand conv end");
                }
                else
                {
                    engine->expandCudnn->Execute(0);
                    if(detailed) Record(events[11],
                        "record SourcePyramid fused expand conv end");
                    const uint32_t c540 = engine->model.Channels540();
                    const size_t shuffleElements=
                        size_t(engine->lowHeight)*engine->lowWidth*c540;
                    const int blocks=int((shuffleElements+threads-1)/threads);
                    if (c540 == 32)
                        PixelShuffle2DequantizeSkipAddHalfKernel<32><<<blocks,threads,0,engine->stream>>>(
                            engine->workAFp8.As<__nv_fp8_e4m3>(),engine->skipFp16.As<Half>(),
                            engine->workBFp16.As<Half>(),int(engine->lowHeight),int(engine->lowWidth),
                            engine->activationScales[expandOutActIdx]);
                    else
                        PixelShuffle2DequantizeSkipAddHalfKernel<48><<<blocks,threads,0,engine->stream>>>(
                            engine->workAFp8.As<__nv_fp8_e4m3>(),engine->skipFp16.As<Half>(),
                            engine->workBFp16.As<Half>(),int(engine->lowHeight),int(engine->lowWidth),
                            engine->activationScales[expandOutActIdx]);
                }
            }
            else
            {
                engine->expandCudnn->Execute(0);
                if(detailed) Record(events[11],
                    "record SourcePyramid fused expand conv end");
                const uint32_t c540 = engine->model.Channels540();
                const size_t shuffleElements=
                    size_t(engine->lowHeight)*engine->lowWidth*c540;
                const int blocks=int((shuffleElements+threads-1)/threads);
                if (c540 == 32)
                    PixelShuffle2SkipAddHalfKernel<32><<<
                        blocks,threads,0,engine->stream>>>(
                        engine->workAFp16.As<Half>(),engine->skipFp16.As<Half>(),
                        engine->workBFp16.As<Half>(),int(engine->lowHeight),
                        int(engine->lowWidth));
                else
                    PixelShuffle2SkipAddHalfKernel<48><<<
                        blocks,threads,0,engine->stream>>>(
                        engine->workAFp16.As<Half>(),engine->skipFp16.As<Half>(),
                        engine->workBFp16.As<Half>(),int(engine->lowHeight),
                        int(engine->lowWidth));
            }
            CheckLaunch("SourcePyramid fused expand epilogue/pixel-shuffle2/skip-add");
            if(detailed)
            {
                Record(events[12],"record SourcePyramid fused expand epilogue end");
                Record(events[13],"record SourcePyramid fused skip-add end");
            }

            // FP8 keeps the cuDNN refine path because its activations remain
            // FP8 (or are written directly to the packed FP8 buffer), while
            // the WMMA kernel consumes the FP16 refine input allocated by the
            // Mixed/FP16 configurations.  Launching it for FP8 would pass
            // the intentionally-null workBFp16 buffer to CUDA.
            const size_t refineLayerIdx = engine->model.Layers().size() - 2;
            const size_t tailLayerIdx = engine->model.Layers().size() - 1;
            if (engine->hasFusedRefineTail &&
                configuredPrecision != RuntimePrecision::Fp8)
            {
                LaunchFusedNativeFp16RefineTailConv(
                    engine->stream,
                    engine->packedWidth,
                    engine->packedHeight,
                    engine->workBFp16.Get(),
                    engine->tailFp16.Get(),
                    engine->refineWeightsPacked.Get(),
                    engine->biases[refineLayerIdx].As<float>(),
                    engine->prelus[refineLayerIdx].As<float>(),
                    engine->tailWeightsPacked.Get(),
                    engine->tailBiasFp16.Get());
                CheckLaunch("SourcePyramid Fused Native SM120 FP16 Refine + Tail");
                if (detailed)
                {
                    Record(events[14],
                        "record SourcePyramid fused refine end");
                    Record(events[15],
                        "record SourcePyramid refine epilogue end");
                }
            }
            else if (engine->hasFastWmmaRefine &&
                configuredPrecision != RuntimePrecision::Fp8)
            {
                LaunchNativeFp16RefineConv(
                    engine->stream,
                    engine->packedWidth,
                    engine->packedHeight,
                    engine->workBFp16.Get(),
                    engine->workAFp16.Get(),
                    engine->refineWeightsPacked.Get(),
                    engine->biases[refineLayerIdx].As<float>(),
                    engine->prelus[refineLayerIdx].As<float>());
                CheckLaunch("SourcePyramid Native SM120 FP16 Refine");
                if (detailed)
                {
                    Record(events[14],
                        "record SourcePyramid fused refine end");
                    Record(events[15],
                        "record SourcePyramid refine epilogue end");
                }
            }
            else if (configuredPrecision == RuntimePrecision::Fp8)
            {
                engine->refineCudnn->Execute(0);
                if (detailed)
                {
                    Record(events[14],
                        "record SourcePyramid fused refine end");
                    Record(events[15],
                        "record SourcePyramid refine epilogue end");
                }
            }
            else
            {
                if (engine->model.AdaptivePairMoE())
                {
                    const dim3 rBlock(32, 8);
                    const dim3 rGrid((engine->packedWidth + 15) / 16, (engine->packedHeight + 7) / 8);
                    RefineWmmaPairMoEKernel<<<rGrid, rBlock, 0, engine->stream>>>(
                        engine->workBFp16.As<Half>(),
                        engine->weightsFp16[refineLayerIdx].As<Half>(),
                        engine->biases[refineLayerIdx].As<float>(),
                        engine->routeThresholdDevice.As<float>(),
                        engine->workAFp16.As<Half>(),
                        engine->routeMetaFp16.As<Half>(),
                        int(engine->packedWidth),
                        int(engine->packedHeight));
                    CheckLaunch("SourcePyramid WMMA FP16 Pair-MoE Refine");
                }
                else
                {
                    engine->refineCudnn->Execute(0);
                }
                if (detailed) Record(events[14],
                    "record SourcePyramid fused refine end");
                if (detailed) Record(events[15],
                    "record SourcePyramid refine epilogue end");
            }

            const dim3 tailBlock(16,16);
            const dim3 tailGrid(
                (engine->outputWidth+tailBlock.x-1)/tailBlock.x,
                (engine->outputHeight+tailBlock.y-1)/tailBlock.y);
            if (!(engine->hasFusedRefineTail && configuredPrecision != RuntimePrecision::Fp8))
            {
                if (engine->hasTailCublasLt)
                {
                    float alpha = 1.0f, beta = 0.0f;
                    CheckCublas(cublasLtMatmul(
                        engine->cublasLt,
                        engine->tailMatmulDesc,
                        &alpha,
                        engine->weightsFp16[tailLayerIdx].Get(),
                        engine->tailAdesc,
                        engine->workAFp16.Get(),
                        engine->tailBdesc,
                        &beta,
                        nullptr,
                        engine->tailCdesc,
                        engine->tailFp16.Get(),
                        engine->tailDdesc,
                        &engine->tailHeuristicResult.algo,
                        engine->cublasWorkspace.Get(),
                        engine->cublasWorkspace.Bytes(),
                        engine->stream),
                        "cublasLtMatmul(tail)");
                }
                else
                {
                    engine->tailCudnn->Execute(0);
                }
                if (eagerValidation) CheckCuda(cudaStreamSynchronize(engine->stream),"validate Tensor Core head");
            }
            if (engine->model.AdaptiveFilter())
            {
                if (detailed) Record(events[17],"record adaptive metadata head end");
                const dim3 block(8,32),grid((inputWidth+7)/8,(inputHeight+31)/32);
                if (engine->model.RgbResidualHead() || engine->model.DepthToSpaceHead())
                {
                    const dim3 directBlock(16,8),directGrid((inputWidth+15)/16,(inputHeight+7)/8);
                    if (engine->model.RgbResidualHead())
                        aess::adaptive::ReconstructDirectRgb<true><<<directGrid,directBlock,0,engine->stream>>>(
                            externalInput.Surface(),engine->tailFp16.As<__half>(),engine->adaptiveLearnedBank,
                            externalOutput.Surface(),int(inputWidth),int(inputHeight),int(engine->packedWidth),
                            int(engine->sourcePadX),int(engine->sourcePadY));
                    else
                        aess::adaptive::ReconstructDirectRgb<false><<<directGrid,directBlock,0,engine->stream>>>(
                            externalInput.Surface(),engine->tailFp16.As<__half>(),engine->adaptiveLearnedBank,
                            externalOutput.Surface(),int(inputWidth),int(inputHeight),int(engine->packedWidth),
                            int(engine->sourcePadX),int(engine->sourcePadY));
                }
                else if (engine->model.AdaptivePairMoE())
                {
                    aess::adaptive::ReconstructPairMoEConstant<8,32><<<grid,block,0,engine->stream>>>(externalInput.Surface(),
                        engine->tailFp16.As<__half>(),engine->routeMetaFp16.As<__half>(),
                        engine->adaptiveLearnedMoEBank,externalOutput.Surface(),
                        int(inputWidth),int(inputHeight),int(engine->packedWidth),
                        int(engine->sourcePadX),int(engine->sourcePadY));
                }
                else if (engine->model.AdaptiveOc25())
                {
                    const dim3 block16_8(16,8),grid16_8((inputWidth+15)/16,(inputHeight+7)/8);
                    const __half* reconstructionMetadata = engine->tailFp16.As<__half>();
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
                    if (contextSettings.Active())
                    {
                        CheckCuda(aess::adaptive::context::Launch(reconstructionMetadata,contextMetadata.As<__half>(),
                            int(inputWidth),int(inputHeight),int(engine->packedWidth),int(engine->packedHeight),
                            int(engine->sourcePadX),int(engine->sourcePadY),contextSettings.parameters,engine->stream),
                            "coefficient context");
                        reconstructionMetadata = contextMetadata.As<__half>();
                    }
#endif
                    static const bool s_useFp16Recon = []() {
                        const char* env = std::getenv("AESS_FP16_RECON");
                        return env && std::atoi(env) != 0;
                    }();
                    if (s_useFp16Recon)
                    {
                        aess::adaptive::ReconstructConstantHalf<16,8><<<grid16_8,block16_8,0,engine->stream>>>(externalInput.Surface(),
                            reconstructionMetadata,engine->adaptiveLearnedBankHalf,externalOutput.Surface(),
                            int(inputWidth),int(inputHeight),int(engine->packedWidth),
                            int(engine->sourcePadX),int(engine->sourcePadY));
                    }
                    else
                    {
                        aess::adaptive::ReconstructConstant<16,8,true,true><<<grid16_8,block16_8,0,engine->stream>>>(externalInput.Surface(),
                            reconstructionMetadata,engine->adaptiveLearnedBank,externalOutput.Surface(),
                            int(inputWidth),int(inputHeight),int(engine->packedWidth),
                            int(engine->sourcePadX),int(engine->sourcePadY));
                    }
                }
                else if (engine->model.AdaptiveFeature8())
                {
                    const dim3 block16_8(16,8),grid16_8((inputWidth+15)/16,(inputHeight+7)/8);
                    aess::adaptive::ReconstructConstant<16,8,true><<<grid16_8,block16_8,0,engine->stream>>>(externalInput.Surface(),
                        engine->tailFp16.As<__half>(),engine->adaptiveBasisBank,externalOutput.Surface(),
                        int(inputWidth),int(inputHeight),int(engine->packedWidth),
                        int(engine->sourcePadX),int(engine->sourcePadY));
                }
                else
                {
                    const dim3 block16_8(16,8),grid16_8((inputWidth+15)/16,(inputHeight+7)/8);
                    aess::adaptive::ReconstructConstant<16,8,false><<<grid16_8,block16_8,0,engine->stream>>>(externalInput.Surface(),
                        engine->tailFp16.As<__half>(),engine->adaptiveBasisBank,externalOutput.Surface(),
                        int(inputWidth),int(inputHeight),int(engine->packedWidth),
                        int(engine->sourcePadX),int(engine->sourcePadY));
                }
            }
            else
            {
                ComposeTailHalfKernel<<<tailGrid,tailBlock,0,engine->stream>>>(
                    engine->tailFp16.As<Half>(),engine->edge.As<ushort4>(),
                    int(engine->packedWidth),int(engine->outputWidth),
                    int(engine->outputHeight),engine->model.EdgePhaseMaxShift(),
                    externalOutput.Surface());
            }
            CheckLaunch("SourcePyramid Tensor Core tail/fused compose");
            Record(events[16],"record SourcePyramid frame end");

            cudaExternalSemaphoreSignalParams signal{};
            signal.params.fence.value = outputReadyValue;
            CheckCuda(cudaSignalExternalSemaphoresAsync(
                &externalSemaphore,&signal,1,engine->stream),
                "cudaSignalExternalSemaphoresAsync(SourcePyramid capture)");
            if (detailed) Record(ownershipTimingEvent,
                "record SourcePyramid ownership transfer");
            Record(completionEvent,"record SourcePyramid completion");
        }

        GraphVariant CaptureGraph(
            bool detailed,
            RuntimeOutputMode outputMode)
        {
            GraphVariant result{};
            CheckCuda(cudaStreamBeginCapture(
                engine->stream,cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture(SourcePyramid)");
            bool active = true;
            try
            {
                EnqueueGraphBody(detailed,outputMode,0,1);
                CheckCuda(cudaStreamEndCapture(engine->stream,&result.graph),
                    "cudaStreamEndCapture(SourcePyramid)");
                active = false;
                size_t count = 0;
                CheckCuda(cudaGraphGetNodes(result.graph,nullptr,&count),
                    "cudaGraphGetNodes(SourcePyramid count)");
                std::vector<cudaGraphNode_t> nodes(count);
                CheckCuda(cudaGraphGetNodes(result.graph,nodes.data(),&count),
                    "cudaGraphGetNodes(SourcePyramid list)");
                size_t waits = 0, signals = 0, kernels = 0;
                for (cudaGraphNode_t node : nodes)
                {
                    cudaGraphNodeType type{};
                    CheckCuda(cudaGraphNodeGetType(node,&type),
                        "cudaGraphNodeGetType(SourcePyramid)");
                    if (type == cudaGraphNodeTypeKernel) ++kernels;
                    else if (type == cudaGraphNodeTypeExtSemaphoreWait)
                    {
                        result.waitNode = node;
                        ++waits;
                    }
                    else if (type == cudaGraphNodeTypeExtSemaphoreSignal)
                    {
                        result.signalNode = node;
                        ++signals;
                    }
                }
                size_t expectedKernels = outputMode == RuntimeOutputMode::EasuOnly ? 1u : 11u;
                if (outputMode != RuntimeOutputMode::EasuOnly && engine->hasParallel32x4Bottlenecks)
                {
                    expectedKernels += 2;
                }
                if (outputMode != RuntimeOutputMode::EasuOnly && configuredPrecision == RuntimePrecision::Mixed)
                {
                    if (engine->hasFusedNativeFp8Expand) --expectedKernels;
                    if (engine->hasFusedNativeFp8Compress) --expectedKernels;
                }
                if (outputMode != RuntimeOutputMode::EasuOnly && engine->hasFusedRefineTail &&
                    configuredPrecision != RuntimePrecision::Fp8)
                {
                    --expectedKernels;
                }
                if (outputMode != RuntimeOutputMode::EasuOnly && engine->hasFusedProject &&
                    configuredPrecision != RuntimePrecision::Fp8)
                {
                    --expectedKernels;
                }
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
                if (outputMode != RuntimeOutputMode::EasuOnly && contextSettings.Active()) ++expectedKernels;
#endif
                if (kernels != expectedKernels || waits != 1 || signals != 1)
                    throw std::runtime_error(
                        "SourcePyramid CUDA graph topology does not match its architecture: expected " +
                        std::to_string(expectedKernels) + " kernels, found " +
                        std::to_string(kernels));
                CheckCuda(cudaGraphInstantiate(&result.executable,result.graph,0),
                    "cudaGraphInstantiate(SourcePyramid)");
                return result;
            }
            catch (...)
            {
                if (active)
                {
                    cudaGraph_t abandoned = nullptr;
                    cudaStreamEndCapture(engine->stream,&abandoned);
                    if (abandoned != nullptr) cudaGraphDestroy(abandoned);
                }
                result.Reset();
                throw;
            }
        }

        void BuildGraphs()
        {
            DestroyGraphs();
            if (eagerValidation) return;
            CheckCuda(cudaStreamSynchronize(engine->stream),
                "cudaStreamSynchronize(SourcePyramid pre-capture)");
            try
            {
                graphs[GraphIndex(false,RuntimeOutputMode::Neural)] =
                    CaptureGraph(false,RuntimeOutputMode::Neural);
                graphs[GraphIndex(true,RuntimeOutputMode::Neural)] =
                    CaptureGraph(true,RuntimeOutputMode::Neural);
                if (!engine->model.AdaptiveFilter())
                {
                    graphs[GraphIndex(false,RuntimeOutputMode::EasuOnly)] =
                        CaptureGraph(false,RuntimeOutputMode::EasuOnly);
                    graphs[GraphIndex(true,RuntimeOutputMode::EasuOnly)] =
                        CaptureGraph(true,RuntimeOutputMode::EasuOnly);
                }
                for (GraphVariant& graph : graphs)
                    if (graph.executable)
                        CheckCuda(cudaGraphUpload(graph.executable,engine->stream),
                            "cudaGraphUpload(SourcePyramid)");
                CheckCuda(cudaStreamSynchronize(engine->stream),
                    "cudaStreamSynchronize(SourcePyramid graph upload)");
            }
            catch (...)
            {
                DestroyGraphs();
                throw;
            }
        }

        void LaunchGraph(
            bool detailed,
            RuntimeOutputMode outputMode,
            uint64_t inputValue,
            uint64_t outputValue)
        {
            if (eagerValidation)
            {
                engine->d3dContext4->Flush();
                EnqueueGraphBody(true,outputMode,inputValue,outputValue);
                return;
            }
            GraphVariant& graph = graphs[GraphIndex(detailed,outputMode)];
            if (graph.executable == nullptr || graph.waitNode == nullptr ||
                graph.signalNode == nullptr)
                throw std::runtime_error("SourcePyramid CUDA graph is not configured");
            cudaExternalSemaphoreWaitParams wait{};
            wait.params.fence.value = inputValue;
            cudaExternalSemaphoreWaitNodeParams waitNode{};
            waitNode.extSemArray = &externalSemaphore;
            waitNode.paramsArray = &wait;
            waitNode.numExtSems = 1;
            CheckCuda(cudaGraphExecExternalSemaphoresWaitNodeSetParams(
                graph.executable,graph.waitNode,&waitNode),
                "cudaGraphExecExternalSemaphoresWaitNodeSetParams(SourcePyramid)");
            cudaExternalSemaphoreSignalParams signal{};
            signal.params.fence.value = outputValue;
            cudaExternalSemaphoreSignalNodeParams signalNode{};
            signalNode.extSemArray = &externalSemaphore;
            signalNode.paramsArray = &signal;
            signalNode.numExtSems = 1;
            CheckCuda(cudaGraphExecExternalSemaphoresSignalNodeSetParams(
                graph.executable,graph.signalNode,&signalNode),
                "cudaGraphExecExternalSemaphoresSignalNodeSetParams(SourcePyramid)");
            CheckCuda(cudaGraphLaunch(graph.executable,engine->stream),
                "cudaGraphLaunch(SourcePyramid)");
        }

        void ResetInterop() noexcept
        {
            if (engine && engine->cudaDevice >= 0) cudaSetDevice(engine->cudaDevice);
            if (pending && completionEvent != nullptr)
            {
                cudaEventSynchronize(completionEvent);
                if (engine && engine->d3dContext4 && d3dFence &&
                    pendingOutputFenceValue != 0)
                    engine->d3dContext4->Wait(d3dFence,pendingOutputFenceValue);
                pending = false;
            }
            DestroyGraphs();
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
            contextMetadata.Reset();
#endif
            externalInput.Reset();
            externalOutput.Reset();
            if (externalSemaphore != nullptr)
            {
                cudaDestroyExternalSemaphore(externalSemaphore);
                externalSemaphore = nullptr;
            }
            if (d3dFence != nullptr)
            {
                d3dFence->Release();
                d3dFence = nullptr;
            }
            if (attachedToConfiguration && engine)
            {
                engine->DetachConfiguration();
                attachedToConfiguration = false;
            }
            synchronizationValue = 0;
            pendingOutputFenceValue = 0;
            pendingSubmissionOrdinal = 0;
            inputTexture = nullptr;
            outputTexture = nullptr;
            inputWidth = inputHeight = 0;
            if (completionWakeEvent != nullptr) ResetEvent(completionWakeEvent);
        }

        void Configure(
            ID3D11Device* device,ID3D11Texture2D* newInput,
            ID3D11Texture2D* newOutput,uint32_t width,uint32_t height,
            RuntimePrecision precision)
        {
            if (device == nullptr || newInput == nullptr || newOutput == nullptr ||
                width == 0 || height == 0)
                throw std::invalid_argument(
                    "SourcePyramid Configure received a null resource or zero dimensions");
            engine->EnsureCudaDevice(device);
            engine->EnsureD3DInterop(device);
            EnsureSlotEvents();
            if (inputTexture == newInput && outputTexture == newOutput &&
                inputWidth == width && inputHeight == height &&
                configuredPrecision == precision)
                return;
            ResetInterop();
            D3D11_TEXTURE2D_DESC inputDesc{},outputDesc{};
            newInput->GetDesc(&inputDesc);
            newOutput->GetDesc(&outputDesc);
            if (inputDesc.Width != width || inputDesc.Height != height ||
                inputDesc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
                outputDesc.Width != width*2 || outputDesc.Height != height*2 ||
                outputDesc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT)
                throw std::runtime_error(
                    "SourcePyramid requires RGBA16F input and exact-2x RGBA16F output");
            constexpr UINT sharedFlags = D3D11_RESOURCE_MISC_SHARED |
                D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
            if ((inputDesc.MiscFlags&sharedFlags)==0 ||
                (outputDesc.MiscFlags&sharedFlags)==0)
                throw std::runtime_error(
                    "SourcePyramid requires D3D11 shared textures");
            try
            {
                CreateSlotSynchronization(device);
                externalInput.Import(newInput,width,height);
                externalOutput.Import(newOutput,width*2,height*2);
                engine->AttachConfiguration(width,height,precision);
                attachedToConfiguration = true;
                inputTexture = newInput;
                outputTexture = newOutput;
                inputWidth = width;
                inputHeight = height;
                configuredPrecision = precision;
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
                if (contextSettings.enabled && (!engine->model.AdaptiveOc25() || engine->model.AdaptivePairMoE()))
                    throw std::invalid_argument("Coefficient context requires the 16-basis OC25 AIF architecture");
                if (contextSettings.Active())
                {
                    contextMetadata.Allocate(engine->tailFp16.Bytes());
                    // Only logical source tokens are written; reconstruction never reads the halo.
                    CheckCuda(cudaMemsetAsync(contextMetadata.Get(),0,contextMetadata.Bytes(),engine->stream),
                        "initialize coefficient context padding");
                }
#endif
                BuildGraphs();
            }
            catch (...)
            {
                ResetInterop();
                throw;
            }
        }

        void ProcessAsync(
            RuntimePrecision precision,
            RuntimeOutputMode outputMode)
        {
            if (!attachedToConfiguration || externalInput.Surface()==0 ||
                externalOutput.Surface()==0 || externalSemaphore==nullptr ||
                engine->d3dContext4==nullptr || d3dFence==nullptr)
                throw std::runtime_error("SourcePyramid CUDA runtime is not configured");
            if (pending)
                throw std::runtime_error("SourcePyramid runtime already has a pending frame");
            if (precision != configuredPrecision)
                throw std::runtime_error(
                    "SourcePyramid precision changed without reconfiguration");
            if (!ResetEvent(completionWakeEvent))
                throw std::runtime_error(
                    "SourcePyramid could not reset its completion event");
            const auto begin = std::chrono::steady_clock::now();
            pendingTimings = {};
            pendingDetailed = timingMode == RuntimeTimingMode::Detailed;
            pendingOutputMode = outputMode;
            pendingTimings.detailed = pendingDetailed;
            bool enqueued = false;
            bool signaled = false;
            bool wakeRegistered = false;
            try
            {
                const uint64_t inputReady = synchronizationValue + 1;
                const HRESULT signalHr = engine->d3dContext4->Signal(
                    d3dFence,inputReady);
                if (FAILED(signalHr))
                    throw std::runtime_error(
                        "ID3D11DeviceContext4::Signal(SourcePyramid input) failed");
                engine->d3dContext4->Flush();
                synchronizationValue = inputReady;
                pendingOutputFenceValue = synchronizationValue + 1;
                const auto launchBegin = std::chrono::steady_clock::now();
                LaunchGraph(
                    pendingDetailed,
                    pendingOutputMode,
                    inputReady,
                    pendingOutputFenceValue);
                enqueued = true;
                synchronizationValue = pendingOutputFenceValue;
                signaled = true;
                pendingTimings.interopMapMs = float(
                    std::chrono::duration<double,std::milli>(launchBegin-begin).count());
                const HRESULT completionHr = d3dFence->SetEventOnCompletion(
                    pendingOutputFenceValue,completionWakeEvent);
                if (FAILED(completionHr))
                    throw std::runtime_error(
                        "ID3D11Fence::SetEventOnCompletion(SourcePyramid) failed");
                wakeRegistered = true;
                pendingTimings.interopUnmapMs = float(
                    std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-launchBegin).count());
                pendingSubmissionOrdinal = engine->CommitSubmission();
                pending = true;
            }
            catch (...)
            {
                if (enqueued) cudaStreamSynchronize(engine->stream);
                if (signaled && pendingOutputFenceValue != 0)
                    engine->d3dContext4->Wait(d3dFence,pendingOutputFenceValue);
                if (wakeRegistered)
                    WaitForSingleObject(completionWakeEvent,INFINITE);
                pendingOutputFenceValue = 0;
                pendingSubmissionOrdinal = 0;
                pending = false;
                ResetEvent(completionWakeEvent);
                throw;
            }
        }

        FrameTimings CollectCompleted()
        {
            pendingTimings.totalCudaMs = EventElapsed(events[0],events[16]);
            if (pendingDetailed)
            {
                pendingTimings.easuPackMs = EventElapsed(events[0],events[1]);
                if (pendingOutputMode == RuntimeOutputMode::EasuOnly)
                {
                    pendingTimings.interopUnmapMs +=
                        EventElapsed(events[16],ownershipTimingEvent);
                }
                else
                {
                    pendingTimings.bodyLayerMs[0] =
                        EventElapsed(events[1],events[3]);
                    pendingTimings.bodyLayerMs[1] =
                        EventElapsed(events[4],events[10]);
                    pendingTimings.bodyLayerMs[2] =
                        EventElapsed(events[10],events[15]);
                    const int convBegin[6] = {1,4,6,8,10,13};
                    const int convEnd[6] = {2,5,7,9,11,14};
                    const int epiEnd[6] = {3,6,8,10,12,15};
                    for (size_t layer=0;layer<6;++layer)
                    {
                        pendingTimings.bodyConvolutionMs += EventElapsed(
                            events[convBegin[layer]],events[convEnd[layer]]);
                        pendingTimings.bodyEpilogueMs += EventElapsed(
                            events[convEnd[layer]],events[epiEnd[layer]]);
                    }
                    pendingTimings.tailConversionMs =
                        EventElapsed(events[3],events[4])+
                        EventElapsed(events[12],events[13]);
                    pendingTimings.tailGemmMs = engine->model.AdaptiveFilter()
                        ? EventElapsed(events[15],events[17])
                        : EventElapsed(events[15],events[16]);
                    if (engine->model.AdaptiveFilter())
                        pendingTimings.reconstructionMs=EventElapsed(events[17],events[16]);
                    // SourcePyramid uses this otherwise-unused field to expose
                    // the expand + pixel-shuffle/skip-add group independently
                    // from the following full-resolution refine convolution.
                    pendingTimings.compositionMs =
                        EventElapsed(events[10],events[13]);
                    pendingTimings.interopUnmapMs +=
                        EventElapsed(events[16],ownershipTimingEvent);
                }
            }
            const HRESULT hr = engine->d3dContext4->Wait(
                d3dFence,pendingOutputFenceValue);
            if (FAILED(hr))
                throw std::runtime_error(
                    "ID3D11DeviceContext4::Wait(SourcePyramid) failed");
            pending = false;
            return pendingTimings;
        }

        bool TryCollect(FrameTimings& timings)
        {
            if (!pending) return false;
            if (d3dFence->GetCompletedValue() < pendingOutputFenceValue)
                return false;
            if (pendingDetailed)
                CheckCuda(cudaEventSynchronize(completionEvent),
                    "cudaEventSynchronize(SourcePyramid detailed)");
            timings = CollectCompleted();
            return true;
        }

        FrameTimings Wait()
        {
            if (!pending)
                throw std::runtime_error("SourcePyramid has no pending frame");
            CheckCuda(cudaEventSynchronize(completionEvent),
                "cudaEventSynchronize(SourcePyramid frame)");
            return CollectCompleted();
        }

        void WriteDebugTensors(const std::filesystem::path& directory) const
        {
            if (pending)
                throw std::runtime_error(
                    "SourcePyramid cannot dump tensors while a frame is pending");
            std::filesystem::create_directories(directory);
            auto write = [&directory](const char* name,const DeviceBuffer& source)
            {
                std::vector<std::byte> host(source.Bytes());
                if (!host.empty())
                    CheckCuda(cudaMemcpy(
                        host.data(),source.Get(),host.size(),cudaMemcpyDeviceToHost),
                        "cudaMemcpy(SourcePyramid debug tensor)");
                std::ofstream output(directory/name,std::ios::binary);
                output.write(reinterpret_cast<const char*>(host.data()),
                    std::streamsize(host.size()));
            };
            if (configuredPrecision == RuntimePrecision::Fp8)
            {
                write("packed.bin",engine->packedFp8);
                if (engine->model.AdaptiveFilter())
                    write("refine_half.bin",engine->packedFp16);
                write("skip.bin",engine->skipFp8);
                write("low_a.bin",engine->lowAFp8);
                write("low_b.bin",engine->lowBFp8);
                write("work_a.bin",engine->workAFp8);
                write("work_b.bin",engine->workBFp8);
            }
            else if (configuredPrecision == RuntimePrecision::Mixed)
            {
                write("packed.bin",engine->packedFp16);
                write("skip.bin",engine->skipFp16);
                write("low_a.bin",engine->lowAFp8);
                write("low_b.bin",engine->lowBFp8);
                write("work_a.bin",engine->workAFp8);
                write("work_b.bin",engine->workBFp16);
            }
            else
            {
                write("packed.bin",engine->packedFp16);
                write("skip.bin",engine->skipFp16);
                write("low_a.bin",engine->lowAFp16);
                write("low_b.bin",engine->lowBFp16);
                write("work_a.bin",engine->workAFp16);
                write("work_b.bin",engine->workBFp16);
            }
            write("edge.bin",engine->edge);
            write("tail.bin",engine->tailFp16);
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
            if (contextSettings.Active()) write("context_tail.bin",contextMetadata);
#endif
            write("accumulator.bin",engine->accumulator);
            std::ofstream metadata(directory/"metadata.txt");
            metadata << engine->packedWidth << ' ' << engine->packedHeight << ' '
                << engine->lowWidth << ' ' << engine->lowHeight << ' '
                << engine->outputWidth << ' ' << engine->outputHeight << '\n';
            for (float scale : engine->activationScales) metadata << scale << ' ';
            metadata << '\n';
        }

        std::shared_ptr<SharedExecutionEngine> engine;
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
        aess::adaptive::context::Settings contextSettings = aess::adaptive::context::ReadEnvironment();
        DeviceBuffer contextMetadata;
#endif
        // Diagnostic only: isolate asynchronous CUDA failures to an exact
        // stage. The default production path remains a single graph launch.
        bool eagerValidation = [] {
            char value[8]{};
            return GetEnvironmentVariableA("AESS_NEURAL_EAGER_VALIDATE",value,8) == 1 && value[0] == '1';
        }();
        std::array<GraphVariant,4> graphs{};
        std::array<cudaEvent_t,18> events{};
        cudaEvent_t ownershipTimingEvent = nullptr;
        cudaEvent_t completionEvent = nullptr;
        HANDLE completionWakeEvent = nullptr;
        FrameTimings pendingTimings{};
        RuntimeTimingMode timingMode = RuntimeTimingMode::TotalOnly;
        RuntimeOutputMode pendingOutputMode = RuntimeOutputMode::Neural;
        bool pendingDetailed = false;
        bool pending = false;
        ExternalD3D11Resource externalInput;
        ExternalD3D11Resource externalOutput;
        cudaExternalSemaphore_t externalSemaphore = nullptr;
        ID3D11Fence* d3dFence = nullptr;
        uint64_t synchronizationValue = 0;
        uint64_t pendingOutputFenceValue = 0;
        uint64_t pendingSubmissionOrdinal = 0;
        bool attachedToConfiguration = false;
        ID3D11Texture2D* inputTexture = nullptr;
        ID3D11Texture2D* outputTexture = nullptr;
        uint32_t inputWidth = 0;
        uint32_t inputHeight = 0;
        RuntimePrecision configuredPrecision = RuntimePrecision::Fp8;
    };

    CudaRuntime::CudaRuntime(const std::filesystem::path& modelPath)
        : m_impl(std::make_unique<Impl>(modelPath))
    {
    }

    CudaRuntime::CudaRuntime(std::unique_ptr<Impl> impl) noexcept
        : m_impl(std::move(impl))
    {
    }

    CudaRuntime::~CudaRuntime() = default;

    std::unique_ptr<CudaRuntime> CudaRuntime::CreateSibling() const
    {
        return std::unique_ptr<CudaRuntime>(
            new CudaRuntime(std::make_unique<Impl>(m_impl->engine)));
    }

    bool CudaRuntime::SharesExecutionEngineWith(
        const CudaRuntime& other) const noexcept
    {
        return m_impl->engine.get() == other.m_impl->engine.get();
    }

    void CudaRuntime::Configure(
        ID3D11Device* device,ID3D11Texture2D* inputTexture,
        ID3D11Texture2D* outputTexture,uint32_t inputWidth,
        uint32_t inputHeight,RuntimePrecision precision)
    {
        m_impl->Configure(
            device,inputTexture,outputTexture,inputWidth,inputHeight,precision);
    }

    void CudaRuntime::Submit(
        RuntimePrecision precision,
        RuntimeOutputMode outputMode)
    {
        m_impl->ProcessAsync(precision,outputMode);
    }

    void CudaRuntime::ProcessAsyncAfter(
        RuntimePrecision precision,
        const CudaRuntime* dependency,
        RuntimeOutputMode outputMode)
    {
        if (dependency != nullptr)
        {
            if (!SharesExecutionEngineWith(*dependency))
                throw std::invalid_argument(
                    "SourcePyramid dependency must share its execution engine");
            if (!dependency->m_impl->pending)
                throw std::invalid_argument(
                    "SourcePyramid dependency is not pending");
            if (dependency->m_impl->pendingSubmissionOrdinal !=
                m_impl->engine->lastSubmissionOrdinal)
                throw std::invalid_argument(
                    "SourcePyramid dependency is not the latest FIFO submission");
        }
        m_impl->ProcessAsync(precision,outputMode);
    }

    bool CudaRuntime::TryCollect(FrameTimings& timings)
    {
        return m_impl->TryCollect(timings);
    }

    FrameTimings CudaRuntime::Wait() { return m_impl->Wait(); }
    FrameTimings CudaRuntime::Process(
        RuntimePrecision precision,
        RuntimeOutputMode outputMode)
    {
        Submit(precision,outputMode);
        return Wait();
    }
    bool CudaRuntime::IsPending() const noexcept { return m_impl->pending; }
    HANDLE CudaRuntime::CompletionEvent() const noexcept
    {
        return m_impl->completionWakeEvent;
    }
    void* CudaRuntime::StreamHandle() const noexcept
    {
        return m_impl->engine->stream;
    }
    uint64_t CudaRuntime::InputFenceValue() const noexcept
    {
        return m_impl->synchronizationValue > 0
            ? m_impl->synchronizationValue-1 : 0;
    }
    uint64_t CudaRuntime::OutputFenceValue() const noexcept
    {
        return m_impl->pendingOutputFenceValue;
    }
    void CudaRuntime::SetTimingMode(RuntimeTimingMode mode)
    {
        if (m_impl->pending)
            throw std::runtime_error(
                "SourcePyramid timing mode cannot change while pending");
        m_impl->timingMode = mode;
    }
    RuntimeTimingMode CudaRuntime::TimingMode() const noexcept
    {
        return m_impl->timingMode;
    }
    void CudaRuntime::ResetInterop() noexcept { m_impl->ResetInterop(); }
    const std::string& CudaRuntime::StartupReport() const noexcept
    {
        return m_impl->engine->startupReport;
    }
    uint64_t CudaRuntime::PersistentDeviceBytes() const noexcept
    {
        return m_impl->engine->persistentBytes
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
            + m_impl->contextMetadata.Bytes()
#endif
            ;
    }
    bool CudaRuntime::AdaptiveFilter() const noexcept
    {
        return m_impl->engine->model.AdaptiveFilter();
    }
    bool CudaRuntime::AdaptiveFeature8() const noexcept
    {
        return m_impl->engine->model.AdaptiveFeature8();
    }
    bool CudaRuntime::AdaptivePairMoE() const noexcept
    {
        return m_impl->engine->model.AdaptivePairMoE();
    }
    bool CudaRuntime::AdaptiveParallel32x4() const noexcept
    {
        return m_impl->engine ? m_impl->engine->hasParallel32x4Bottlenecks : false;
    }
    bool CudaRuntime::HasNvFp4() const noexcept
    {
        return m_impl->engine ? m_impl->engine->hasNvFp4Bottlenecks : false;
    }

    uint32_t CudaRuntime::InputWidth() const noexcept
    {
        return m_impl->inputWidth;
    }
    uint32_t CudaRuntime::InputHeight() const noexcept
    {
        return m_impl->inputHeight;
    }
    void CudaRuntime::WriteDebugTensors(
        const std::filesystem::path& directory) const
    {
        m_impl->WriteDebugTensors(directory);
    }
}
