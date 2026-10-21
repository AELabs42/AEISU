#pragma once

#include <array>
#include <cstddef>

namespace aess::neural
{
    // The presenter profiles three coarse neural execution groups. Keeping
    // this transport shape model-neutral lets the active runtime change
    // without making the presentation pipeline depend on a retired model.
    inline constexpr std::size_t kProfiledLayerGroupCount = 3;

    enum class RuntimePrecision
    {
        Fp16,
        Fp8,
        Mixed,
    };

    // TotalOnly is the production default. Detailed timing is opt-in and is
    // reserved for profiling so normal presentation records only the minimum
    // CUDA event set.
    enum class RuntimeTimingMode
    {
        TotalOnly,
        Detailed,
    };

    // EasuOnly publishes SourcePyramid's real spatial base while bypassing
    // the neural body. It is used only by the debug comparison view.
    enum class RuntimeOutputMode
    {
        Neural,
        EasuOnly,
    };

    struct FrameTimings
    {
        float interopMapMs = 0.0f;
        float easuPackMs = 0.0f;
        std::array<float, kProfiledLayerGroupCount> bodyLayerMs{};
        float bodyConvolutionMs = 0.0f;
        float bodyEpilogueMs = 0.0f;
        float tailConversionMs = 0.0f;
        float tailGemmMs = 0.0f;
        float compositionMs = 0.0f;
        float reconstructionMs = 0.0f; // Adaptive output only, detailed timing.
        float interopUnmapMs = 0.0f;
        float totalCudaMs = 0.0f;
        bool detailed = false;
    };
}
