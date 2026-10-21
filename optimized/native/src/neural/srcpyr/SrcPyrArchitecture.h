#pragma once

#include <cstddef>
#include <cstdint>

namespace aess::srcpyr
{
    // Versioned on-disk contract for the source-pyramid model.  Exporters
    // should convert the training checkpoint into this fixed, endian-tagged
    // container before the native runtime consumes it.
    inline constexpr uint32_t kArtifactFormatVersion = 1;
    inline constexpr char kArchitectureId[] =
        "srcpyr48_64_targetmeta_pack80_edgephase_v6";
    inline constexpr char kAdaptiveArchitectureId[] =
        "aif16x5_rgby_s48_64_v1opt";
    inline constexpr char kAdaptiveFeature8ArchitectureId[] =
        "aif16x5_feat8_s32_64_v1";
    inline constexpr char kAdaptiveOc25ArchitectureId[] =
        "aif16x5_feat8_s32_64_l5base_normbasis_oc25_v1";
    inline constexpr char kAdaptiveLearnedStencilArchitectureId[] =
        "aif16x5_learnedstencil8_s32_64_learnedbase_normbasis_oc25_v1";
    inline constexpr char kAdaptiveLearnedRgb4DirArchitectureId[] =
        "aif16x5_learnedrgb4dir8_s32_64_learnedbase_normbasis_oc25_v1";
    inline constexpr char kAdaptivePairMoELatentRouteArchitectureId[] =
        "aif16x2x5_luma1_s32_64_fixedbilinear_latentroute_oc25_v1";
    inline constexpr char kAdaptiveParallel32x4ArchitectureId[] =
        "aif16x5_learnedrgb4dir8_s32_parallel32x4_learnedbase_oc25_v1";
    inline constexpr char kRgbResidualArchitectureId[] =
        "aif_rgb_residual_parallel32x4_learnedbase_540to4k_v1";
    inline constexpr char kDepthToSpaceArchitectureId[] =
        "aif_depth_to_space_parallel32x4_540to4k_v1";

    struct LearnedAnalysisStencil
    {
        float rgbMix[3] = {0.2126f, 0.7152f, 0.0722f};
        float dxRows[3] = {0.125f, 0.25f, 0.125f};
        float dyCols[3] = {0.125f, 0.25f, 0.125f};
        float lap[2] = {1.0f, -0.25f};
    };
    static_assert(sizeof(LearnedAnalysisStencil) == 11 * sizeof(float));

    struct LearnedAnalysis4Dir
    {
        float rgbBasis[9]; // 3x3 row-major
        float rgbMix[3];   // luma mix
        float dxRows[3];
        float dyCols[3];
        float diag45[3];
        float diag135[3];
    };
    static_assert(sizeof(LearnedAnalysis4Dir) == 24 * sizeof(float));
    inline constexpr uint32_t kLogicalInputChannels = 20;
    inline constexpr uint32_t kPackedInputChannels = 80;
    inline constexpr uint32_t kChannels540 = 48;
    inline constexpr uint32_t kChannels270 = 64;
    inline constexpr uint32_t kBottleneckLayers = 2;
    inline constexpr uint32_t kParallel32x4BottleneckLayers = 4;
    inline constexpr uint32_t kTailChannels = 64;
    inline constexpr uint32_t kTailOutputChannels = 16;
    inline constexpr uint32_t kActivationScaleCount = 8;
    inline constexpr uint32_t kParallel32x4ActivationScaleCount = 10;
    inline constexpr uint32_t kParallel32x4LayerCount = 9;
    inline constexpr uint32_t kUpscaleFactor = 2;
    inline constexpr uint32_t kTensorAlignment = 64;
    inline constexpr float kDefaultEdgePhaseMaxShift = 1.25f;
    inline constexpr float kDefaultEdgeGateGain = 2.0f;

    enum class TensorDataType : uint32_t
    {
        Fp8E4M3Fn = 1,
        Fp16 = 2,
        Fp32 = 3,
        NvFp4E2M1 = 4,
        NvFp4Block16 = 5,
    };

    enum class TensorLayout : uint32_t
    {
        Scalar = 0,
        Vector = 1,
        Oihw = 2, // PyTorch OIHW, used by the exporter and FP16 fallback.
        Krsc = 3, // CUTLASS K/R/S/C, used by the FP8 convolution path.
        AdaptiveBasis = 4, // [25 taps, 16 coefficients, 4 output phases].
    };

    inline constexpr size_t kMaxTensorName = 32;
    inline constexpr size_t kHeaderSize = 512;
    inline constexpr size_t kDescriptorSize = 80;

    // Canonical names consumed by the runtime.  An exporter may include both
    // `.weight` (FP8/KRSC) and `.weight_fp16` (FP16/OIHW or KRSC) records.
    inline constexpr const char* kLayerNames[] = {
        "project", "compress", "bottleneck0", "bottleneck1", "expand", "refine", "tail"};
    inline constexpr const char* kParallel32x4LayerNames[] = {
        "project", "compress", "bottleneck0", "bottleneck1", "bottleneck2", "bottleneck3", "expand", "refine", "tail"};
}
