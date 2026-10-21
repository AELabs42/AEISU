#pragma once

#include "SrcPyrArchitecture.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace aess::srcpyr
{
    struct TensorView
    {
        TensorDataType dataType{};
        TensorLayout layout{};
        uint32_t rank = 0;
        std::array<uint32_t, 4> shape{};
        const std::byte* data = nullptr;
        uint64_t byteCount = 0;
    };

    struct LayerView
    {
        TensorView weight;
        TensorView weightFp16;
        TensorView weightScale;
        TensorView bias;
        TensorView biasFp8;
        TensorView prelu;
        TensorView weightNvFp4;
        TensorView weightNvFp4Scale;
        bool hasNvFp4 = false;
    };

    // A deliberately small, immutable view of the exported checkpoint.  The
    // backing bytes remain owned by ModelArtifact so CUDA upload can happen
    // without temporary tensor copies.
    class ModelArtifact
    {
    public:
        static ModelArtifact Load(const std::filesystem::path& path);

        const std::filesystem::path& Path() const noexcept { return m_path; }
        const std::string& ChecksumHex() const noexcept { return m_checksumHex; }
        bool AdaptiveFilter() const noexcept { return m_adaptiveFilter; }
        bool AdaptiveFeature8() const noexcept { return m_adaptiveFeature8; }
        bool AdaptiveOc25() const noexcept { return m_adaptiveOc25; }
        bool AdaptiveLearnedStencil() const noexcept { return m_adaptiveLearnedStencil; }
        bool AdaptiveLearnedRgb4Dir() const noexcept { return m_adaptiveLearnedRgb4Dir; }
        bool AdaptivePairMoE() const noexcept { return m_adaptivePairMoE; }
        bool AdaptiveParallel32x4() const noexcept { return m_adaptiveParallel32x4; }
        bool RgbResidualHead() const noexcept { return m_rgbResidualHead; }
        bool DepthToSpaceHead() const noexcept { return m_depthToSpaceHead; }
        bool HasNvFp4() const noexcept { return m_layers.size() > 3 && m_layers[2].hasNvFp4 && m_layers[3].hasNvFp4; }
        const char* ArchitectureId() const noexcept
        {
            if (m_rgbResidualHead) return kRgbResidualArchitectureId;
            if (m_depthToSpaceHead) return kDepthToSpaceArchitectureId;
            return m_adaptiveParallel32x4 ? kAdaptiveParallel32x4ArchitectureId :
                (m_adaptivePairMoE ? kAdaptivePairMoELatentRouteArchitectureId :
                (m_adaptiveLearnedRgb4Dir ? kAdaptiveLearnedRgb4DirArchitectureId :
                (m_adaptiveLearnedStencil ? kAdaptiveLearnedStencilArchitectureId :
                (m_adaptiveOc25 ? kAdaptiveOc25ArchitectureId :
                (m_adaptiveFeature8 ? kAdaptiveFeature8ArchitectureId :
                (m_adaptiveFilter ? kAdaptiveArchitectureId : kArchitectureId))))));
        }
        uint32_t PackedInputChannels() const noexcept { return m_adaptivePairMoE ? 4u : (m_adaptiveFeature8 ? 32u : (m_adaptiveFilter ? 16u : kPackedInputChannels)); }
        uint32_t Channels540() const noexcept { return (m_adaptiveFeature8 || m_adaptivePairMoE) ? 32u : kChannels540; }
        uint32_t RefineChannels() const noexcept { return (m_adaptiveFeature8 || m_adaptivePairMoE) ? 64u : kChannels540; }
        uint32_t TailChannels() const noexcept { return kTailChannels; }
        bool Fp8HalfProject() const noexcept { return m_halfProject; }
        bool Fp8HalfHead() const noexcept { return m_halfHead; }
        const TensorView& AdaptiveBasis() const noexcept { return m_adaptiveBasis; }
        const TensorView& AdaptiveBase() const noexcept { return m_adaptiveBase; }
        const TensorView& RouteThreshold() const noexcept { return m_routeThreshold; }
        const LearnedAnalysisStencil& AnalysisStencil() const noexcept { return m_analysisStencil; }
        const LearnedAnalysis4Dir& Analysis4Dir() const noexcept { return m_analysis4Dir; }
        float EdgePhaseMaxShift() const noexcept { return m_edgePhaseMaxShift; }
        float EdgeGateGain() const noexcept { return m_edgeGateGain; }
        const std::vector<LayerView>& Layers() const noexcept { return m_layers; }
        const TensorView& ActivationScales() const noexcept { return m_activationScales; }

    private:
        std::filesystem::path m_path;
        std::string m_checksumHex;
        bool m_adaptiveFilter = false;
        bool m_adaptiveFeature8 = false;
        bool m_adaptiveOc25 = false;
        bool m_adaptiveLearnedStencil = false;
        bool m_adaptiveLearnedRgb4Dir = false;
        bool m_adaptivePairMoE = false;
        bool m_adaptiveParallel32x4 = false;
        bool m_rgbResidualHead = false;
        bool m_depthToSpaceHead = false;
        bool m_halfProject = false;
        bool m_halfHead = false;
        std::vector<std::byte> m_storage;
        std::vector<LayerView> m_layers{};
        TensorView m_activationScales{};
        TensorView m_adaptiveBasis{};
        TensorView m_adaptiveBase{};
        TensorView m_routeThreshold{};
        LearnedAnalysisStencil m_analysisStencil{};
        LearnedAnalysis4Dir m_analysis4Dir{};
        float m_edgePhaseMaxShift = kDefaultEdgePhaseMaxShift;
        float m_edgeGateGain = kDefaultEdgeGateGain;
    };
}
