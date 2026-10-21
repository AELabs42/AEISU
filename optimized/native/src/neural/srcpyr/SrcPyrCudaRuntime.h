#pragma once

#include "SrcPyrArchitecture.h"
#include "..\NeuralRuntimeTypes.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

struct ID3D11Device;
struct ID3D11Texture2D;

namespace aess::srcpyr
{
    using RuntimePrecision = aess::neural::RuntimePrecision;
    using RuntimeTimingMode = aess::neural::RuntimeTimingMode;
    using RuntimeOutputMode = aess::neural::RuntimeOutputMode;
    using FrameTimings = aess::neural::FrameTimings;

    class CudaRuntime
    {
    public:
        explicit CudaRuntime(const std::filesystem::path& modelPath);
        ~CudaRuntime();

        CudaRuntime(const CudaRuntime&) = delete;
        CudaRuntime& operator=(const CudaRuntime&) = delete;

        std::unique_ptr<CudaRuntime> CreateSibling() const;
        bool SharesExecutionEngineWith(const CudaRuntime& other) const noexcept;

        void Configure(
            ID3D11Device* device,
            ID3D11Texture2D* inputTexture,
            ID3D11Texture2D* outputTexture,
            uint32_t inputWidth,
            uint32_t inputHeight,
            RuntimePrecision precision);

        // Submit is the canonical name for the event-driven API.  ProcessAsync
        // is retained as a source-compatible alias for the current presenter.
        void Submit(
            RuntimePrecision precision,
            RuntimeOutputMode outputMode = RuntimeOutputMode::Neural);
        void ProcessAsync(
            RuntimePrecision precision,
            RuntimeOutputMode outputMode = RuntimeOutputMode::Neural)
        {
            Submit(precision, outputMode);
        }
        void ProcessAsyncAfter(
            RuntimePrecision precision,
            const CudaRuntime* dependency,
            RuntimeOutputMode outputMode = RuntimeOutputMode::Neural);
        bool TryCollect(FrameTimings& timings);
        FrameTimings Wait();
        FrameTimings Process(
            RuntimePrecision precision,
            RuntimeOutputMode outputMode = RuntimeOutputMode::Neural);
        bool IsPending() const noexcept;
        HANDLE CompletionEvent() const noexcept;
        void* StreamHandle() const noexcept;
        uint64_t InputFenceValue() const noexcept;
        uint64_t OutputFenceValue() const noexcept;
        void SetTimingMode(RuntimeTimingMode mode);
        RuntimeTimingMode TimingMode() const noexcept;
        void ResetInterop() noexcept;

        const std::string& StartupReport() const noexcept;
        uint64_t PersistentDeviceBytes() const noexcept;
        uint32_t InputWidth() const noexcept;
        bool AdaptiveFilter() const noexcept;
        bool AdaptiveFeature8() const noexcept;
        bool AdaptivePairMoE() const noexcept;
        bool AdaptiveParallel32x4() const noexcept;
        bool HasNvFp4() const noexcept;
        uint32_t InputHeight() const noexcept;

        // Explicit diagnostic path used by the native numerical gate. It is
        // never called by the presenter and adds no per-frame work.
        void WriteDebugTensors(const std::filesystem::path& directory) const;

    private:
        struct Impl;
        explicit CudaRuntime(std::unique_ptr<Impl> impl) noexcept;
        std::unique_ptr<Impl> m_impl;
    };
}
