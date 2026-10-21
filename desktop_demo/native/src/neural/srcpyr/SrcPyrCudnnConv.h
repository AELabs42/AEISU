#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aess::srcpyr
{
    enum class CudnnConvDataType : uint8_t
    {
        Fp8E4m3,
        Fp16,
    };

    enum class CudnnConvEpilogue : uint8_t
    {
        Fp8PreluRequant,
        Fp16Bias,
        Fp16Prelu,
    };

    struct CudnnConvPointers
    {
        void* input=nullptr;
        const void* weight=nullptr;
        const float* inputScale=nullptr;
        const float* weightScale=nullptr;
        const float* bias=nullptr;
        const float* prelu=nullptr;
        const float* outputReciprocalScale=nullptr;
        void* output=nullptr;
    };

    struct CudnnConvGeometry
    {
        uint32_t width=0;
        uint32_t height=0;
        uint32_t inputChannels=0;
        uint32_t outputChannels=0;
        uint32_t filter=0;
        uint32_t padding=0;
        const char* cacheName=nullptr;
        CudnnConvEpilogue epilogue=CudnnConvEpilogue::Fp8PreluRequant;
        CudnnConvDataType dataType=CudnnConvDataType::Fp8E4m3;
        bool autotune=false;
    };

    // A persistent cuDNN Frontend FP8/FP16 convolution plan. The operation
    // graph fuses scaling (FP8), bias, PReLU, and output conversion into the
    // Tensor Core convolution, avoiding full activation round trips.
    class CudnnConvRuntime
    {
    public:
        CudnnConvRuntime();
        ~CudnnConvRuntime();
        CudnnConvRuntime(const CudnnConvRuntime&)=delete;
        CudnnConvRuntime& operator=(const CudnnConvRuntime&)=delete;

        void Configure(
            const CudnnConvGeometry& geometry,
            void* cudaStream,
            const std::vector<CudnnConvPointers>& variants);
        void Execute(size_t variant);

        uint64_t WorkspaceBytes() const noexcept;
        uint64_t LibraryVersion() const noexcept;
        uint64_t FrontendVersion() const noexcept;
        const std::string& PlanName() const noexcept;
        bool PlanCacheHit() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
