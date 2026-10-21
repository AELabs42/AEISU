#pragma once

#include "NeuralRuntimeTypes.h"
#include "srcpyr/SrcPyrCudaRuntime.h"
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace aess::neural
{
    enum class ModelKind
    {
        AdaptiveFilter32,
    };

    // The demo exposes exactly one deployed architecture and precision profile.
    class CudaRuntime final
    {
    public:
        CudaRuntime(ModelKind kind, const std::filesystem::path& modelPath)
            : m_kind(kind)
        {
            if (kind != ModelKind::AdaptiveFilter32)
            {
                throw std::invalid_argument("Unsupported neural model kind");
            }
            m_runtime = std::make_unique<aess::srcpyr::CudaRuntime>(modelPath);
            if (!m_runtime->AdaptiveFeature8() || !m_runtime->AdaptiveParallel32x4())
                throw std::invalid_argument("The demo requires the current AEISU parallel32x4 artifact.");
        }

        CudaRuntime(const CudaRuntime&) = delete;
        CudaRuntime& operator=(const CudaRuntime&) = delete;

        std::unique_ptr<CudaRuntime> CreateSibling() const
        {
            return std::unique_ptr<CudaRuntime>(
                new CudaRuntime(m_kind, m_runtime->CreateSibling()));
        }

        bool SharesExecutionEngineWith(const CudaRuntime& other) const noexcept
        {
            return m_runtime->SharesExecutionEngineWith(*other.m_runtime);
        }

        void Configure(
            ID3D11Device* device,
            ID3D11Texture2D* inputTexture,
            ID3D11Texture2D* outputTexture,
            uint32_t inputWidth,
            uint32_t inputHeight,
            RuntimePrecision precision)
        {
            if (precision != RuntimePrecision::Mixed)
                throw std::invalid_argument("The demo supports only the mixed FP8/FP16 profile.");
            unsigned int count = 0;
            int devices[8]{};
            if (cudaD3D11GetDevices(&count, devices, 8, device, cudaD3D11DeviceListAll) != cudaSuccess || count == 0)
                throw std::runtime_error("The demo requires an NVIDIA RTX 50-series D3D11 adapter.");
            for (unsigned int index = 0; index < count; ++index)
            {
                cudaDeviceProp properties{};
                if (cudaGetDeviceProperties(&properties, devices[index]) != cudaSuccess ||
                    properties.major != 12 || properties.minor != 0)
                    throw std::runtime_error("This demo build is RTX 50-series only (Blackwell SM 12.0); RTX 40-series is not supported.");
            }
            m_runtime->Configure(
                device,
                inputTexture,
                outputTexture,
                inputWidth,
                inputHeight,
                precision);
        }

        void ProcessAsync(
            RuntimePrecision precision,
            RuntimeOutputMode outputMode = RuntimeOutputMode::Neural)
        {
            m_runtime->ProcessAsync(precision, outputMode);
        }

        void ProcessAsyncAfter(
            RuntimePrecision precision,
            const CudaRuntime* dependency,
            RuntimeOutputMode outputMode = RuntimeOutputMode::Neural)
        {
            m_runtime->ProcessAsyncAfter(
                precision,
                dependency ? dependency->m_runtime.get() : nullptr,
                outputMode);
        }

        bool TryCollect(FrameTimings& timings)
        {
            return m_runtime->TryCollect(timings);
        }

        FrameTimings Wait()
        {
            return m_runtime->Wait();
        }

        FrameTimings Process(
            RuntimePrecision precision,
            RuntimeOutputMode outputMode = RuntimeOutputMode::Neural)
        {
            return m_runtime->Process(precision, outputMode);
        }

        bool IsPending() const noexcept
        {
            return m_runtime->IsPending();
        }

        HANDLE CompletionEvent() const noexcept
        {
            return m_runtime->CompletionEvent();
        }

        void SetTimingMode(RuntimeTimingMode mode)
        {
            m_runtime->SetTimingMode(mode);
        }

        RuntimeTimingMode TimingMode() const noexcept
        {
            return m_runtime->TimingMode();
        }

        void ResetInterop() noexcept
        {
            m_runtime->ResetInterop();
        }

        const std::string& StartupReport() const noexcept
        {
            return m_runtime->StartupReport();
        }

        uint64_t PersistentDeviceBytes() const noexcept
        {
            return m_runtime->PersistentDeviceBytes();
        }

        bool HasNvFp4() const noexcept
        {
            return m_runtime->HasNvFp4();
        }

        uint32_t InputWidth() const noexcept
        {
            return m_runtime->InputWidth();
        }

        uint32_t InputHeight() const noexcept
        {
            return m_runtime->InputHeight();
        }

        ModelKind Kind() const noexcept
        {
            return m_kind;
        }

    private:
        explicit CudaRuntime(
            ModelKind kind,
            std::unique_ptr<aess::srcpyr::CudaRuntime> runtime) noexcept
            : m_kind(kind), m_runtime(std::move(runtime))
        {
        }

        ModelKind m_kind;
        std::unique_ptr<aess::srcpyr::CudaRuntime> m_runtime;
    };
}
