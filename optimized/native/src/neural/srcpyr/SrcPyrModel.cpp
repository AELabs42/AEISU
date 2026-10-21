#include "SrcPyrModel.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace aess::srcpyr
{
    namespace
    {
        constexpr uint32_t kEndianTag = 0x01020304;
        constexpr uint32_t kFp8E4M3Fn = 1;
        constexpr uint32_t kActivationLayout = 1;
        constexpr uint32_t kWeightLayout = static_cast<uint32_t>(TensorLayout::Oihw);
        constexpr uint32_t kScaleConvention = 1;
        constexpr uint32_t kExpectedTensorCount = 1u + 7u * 5u;
        constexpr size_t kChecksumOffset = 184;
        constexpr size_t kChecksumSize = 32;

#pragma pack(push, 1)
        struct HeaderPrefix
        {
            char magic[8];
            uint32_t version;
            uint32_t headerSize;
            uint32_t endianTag;
            uint32_t flags;
            char architecture[64];
            uint32_t logicalInputChannels;
            uint32_t packedInputChannels;
            uint32_t channels540;
            uint32_t channels270;
            uint32_t bottleneckLayers;
            uint32_t tailChannels;
            uint32_t scale;
            uint32_t fp8Format;
            uint32_t activationLayout;
            uint32_t weightLayout;
            uint32_t scaleConvention;
            uint32_t tensorCount;
            uint32_t alignment;
            uint32_t descriptorSize;
            float edgePhaseMaxShift;
            float edgeGateGain;
            uint64_t fileSize;
            uint64_t tocOffset;
            uint64_t tocBytes;
            uint64_t payloadOffset;
            uint8_t checksum[kChecksumSize];
            uint8_t reserved[kHeaderSize - 216];
        };

        struct TensorDescriptor
        {
            char name[kMaxTensorName];
            uint32_t dataType;
            uint32_t layout;
            uint32_t rank;
            uint32_t shape[4];
            uint64_t offset;
            uint64_t byteCount;
            uint8_t reserved[4];
        };
#pragma pack(pop)

        static_assert(sizeof(HeaderPrefix) == kHeaderSize);
        static_assert(offsetof(HeaderPrefix, checksum) == kChecksumOffset);
        static_assert(sizeof(TensorDescriptor) == kDescriptorSize);

        class AlgorithmHandle
        {
        public:
            AlgorithmHandle()
            {
                NTSTATUS status = BCryptOpenAlgorithmProvider(
                    &m_value, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
                if (!BCRYPT_SUCCESS(status))
                {
                    throw std::runtime_error("BCryptOpenAlgorithmProvider(SHA256) failed");
                }
            }
            ~AlgorithmHandle()
            {
                if (m_value != nullptr) BCryptCloseAlgorithmProvider(m_value, 0);
            }
            operator BCRYPT_ALG_HANDLE() const noexcept { return m_value; }
        private:
            BCRYPT_ALG_HANDLE m_value = nullptr;
        };

        class HashHandle
        {
        public:
            explicit HashHandle(BCRYPT_ALG_HANDLE algorithm)
            {
                NTSTATUS status = BCryptCreateHash(
                    algorithm, &m_value, nullptr, 0, nullptr, 0, 0);
                if (!BCRYPT_SUCCESS(status))
                {
                    throw std::runtime_error("BCryptCreateHash failed");
                }
            }
            ~HashHandle()
            {
                if (m_value != nullptr) BCryptDestroyHash(m_value);
            }
            operator BCRYPT_HASH_HANDLE() const noexcept { return m_value; }
        private:
            BCRYPT_HASH_HANDLE m_value = nullptr;
        };

        std::array<uint8_t, kChecksumSize> Sha256WithZeroedChecksum(
            const std::vector<std::byte>& storage)
        {
            AlgorithmHandle algorithm;
            HashHandle hash(algorithm);
            auto append = [&](size_t offset, size_t bytes)
            {
                if (bytes > static_cast<size_t>(std::numeric_limits<ULONG>::max()))
                {
                    throw std::runtime_error("SPYR8 artifact is too large for BCrypt");
                }
                const NTSTATUS status = BCryptHashData(
                    hash,
                    reinterpret_cast<PUCHAR>(const_cast<std::byte*>(storage.data() + offset)),
                    static_cast<ULONG>(bytes),
                    0);
                if (!BCRYPT_SUCCESS(status)) throw std::runtime_error("BCryptHashData failed");
            };
            append(0, kChecksumOffset);
            std::array<uint8_t, kChecksumSize> zeros{};
            NTSTATUS status = BCryptHashData(
                hash, zeros.data(), static_cast<ULONG>(zeros.size()), 0);
            if (!BCRYPT_SUCCESS(status)) throw std::runtime_error("BCryptHashData(checksum) failed");
            append(kChecksumOffset + kChecksumSize,
                   storage.size() - kChecksumOffset - kChecksumSize);
            std::array<uint8_t, kChecksumSize> digest{};
            status = BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0);
            if (!BCRYPT_SUCCESS(status)) throw std::runtime_error("BCryptFinishHash failed");
            return digest;
        }

        std::string Hex(const uint8_t* bytes, size_t count)
        {
            std::ostringstream stream;
            stream << std::hex << std::setfill('0');
            for (size_t i = 0; i < count; ++i)
                stream << std::setw(2) << static_cast<unsigned>(bytes[i]);
            return stream.str();
        }

        std::string FixedString(const char* text, size_t capacity, const char* field)
        {
            const void* end = std::memchr(text, '\0', capacity);
            if (end == nullptr) throw std::runtime_error(std::string("unterminated SPYR8 ") + field);
            return std::string(text, static_cast<const char*>(end));
        }

        uint64_t ElementSize(TensorDataType type)
        {
            switch (type)
            {
            case TensorDataType::NvFp4E2M1: return 1;
            case TensorDataType::Fp8E4M3Fn: return 1;
            case TensorDataType::NvFp4Block16:
            case TensorDataType::Fp16: return 2;
            case TensorDataType::Fp32: return 4;
            default: throw std::runtime_error("unsupported SPYR8 tensor dtype");
            }
        }

        uint64_t Align(uint64_t value)
        {
            return (value + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment;
        }

        void RequireTensor(
            const std::unordered_map<std::string, TensorView>& tensors,
            const std::string& name,
            TensorDataType type,
            TensorLayout layout,
            std::initializer_list<uint32_t> shape,
            TensorView& destination)
        {
            auto it = tensors.find(name);
            if (it == tensors.end())
                throw std::runtime_error("missing SPYR8 tensor: " + name);
            const TensorView& value = it->second;
            if (value.dataType != type || value.layout != layout || value.rank != shape.size())
                throw std::runtime_error("SPYR8 tensor metadata mismatch: " + name);
            size_t index = 0;
            for (uint32_t expected : shape)
                if (value.shape[index++] != expected)
                    throw std::runtime_error("SPYR8 tensor shape mismatch: " + name);
            destination = value;
        }

        void RequireWeightPair(
            const std::unordered_map<std::string, TensorView>& tensors,
            const std::string& prefix,
            std::initializer_list<uint32_t> shape,
            LayerView& layer)
        {
            RequireTensor(tensors, prefix + ".weight_fp8", TensorDataType::Fp8E4M3Fn,
                          TensorLayout::Oihw, shape, layer.weight);
            RequireTensor(tensors, prefix + ".weight_fp16", TensorDataType::Fp16,
                          TensorLayout::Oihw, shape, layer.weightFp16);
            const uint32_t outChannels = *shape.begin();
            RequireTensor(tensors, prefix + ".weight_scale", TensorDataType::Fp32,
                          TensorLayout::Vector, {outChannels}, layer.weightScale);
            RequireTensor(tensors, prefix + ".bias", TensorDataType::Fp32,
                          TensorLayout::Vector, {outChannels}, layer.bias);
            RequireTensor(tensors, prefix + ".prelu", TensorDataType::Fp32,
                          TensorLayout::Vector, {outChannels}, layer.prelu);
            auto nvfp4It = tensors.find(prefix + ".weight_nvfp4");
            auto scaleIt = tensors.find(prefix + ".weight_nvfp4_scale");
            if (nvfp4It != tensors.end() && scaleIt != tensors.end())
            {
                layer.weightNvFp4 = nvfp4It->second;
                layer.weightNvFp4Scale = scaleIt->second;
                layer.hasNvFp4 = true;
            }
        }
    }

    ModelArtifact ModelArtifact::Load(const std::filesystem::path& path)
    {
        ModelArtifact model;
        model.m_path = std::filesystem::absolute(path);
        std::ifstream input(model.m_path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("could not open SPYR8 model: " + model.m_path.string());
        const std::streamsize size = input.tellg();
        if (size < static_cast<std::streamsize>(kHeaderSize))
            throw std::runtime_error("SPYR8 artifact is smaller than its header");
        input.seekg(0);
        model.m_storage.resize(static_cast<size_t>(size));
        if (!input.read(reinterpret_cast<char*>(model.m_storage.data()), size))
            throw std::runtime_error("failed to read complete SPYR8 artifact");

        const auto* header = reinterpret_cast<const HeaderPrefix*>(model.m_storage.data());
        constexpr char expectedMagic[8] = {'S','P','Y','R','8','F','8','\0'};
        if (std::memcmp(header->magic, expectedMagic, sizeof(expectedMagic)) != 0 ||
            header->version != kArtifactFormatVersion || header->headerSize != kHeaderSize ||
            header->endianTag != kEndianTag || header->flags != 0)
            throw std::runtime_error("unsupported SPYR8 magic/version/endian/flags");
        const std::string architecture = FixedString(header->architecture, sizeof(header->architecture), "architecture");
        model.m_rgbResidualHead = architecture == kRgbResidualArchitectureId;
        model.m_depthToSpaceHead = architecture == kDepthToSpaceArchitectureId;
        model.m_adaptiveParallel32x4 = architecture == kAdaptiveParallel32x4ArchitectureId ||
            model.m_rgbResidualHead || model.m_depthToSpaceHead;
        model.m_adaptivePairMoE = architecture == kAdaptivePairMoELatentRouteArchitectureId;
        model.m_adaptiveLearnedStencil = architecture == kAdaptiveLearnedStencilArchitectureId;
        model.m_adaptiveLearnedRgb4Dir = architecture == kAdaptiveLearnedRgb4DirArchitectureId || model.m_adaptiveParallel32x4;
        model.m_adaptiveOc25 = architecture == kAdaptiveOc25ArchitectureId || model.m_adaptiveLearnedStencil || model.m_adaptiveLearnedRgb4Dir || model.m_adaptivePairMoE;
        model.m_adaptiveFeature8 = architecture == kAdaptiveFeature8ArchitectureId || model.m_adaptiveOc25;
        model.m_adaptiveFilter = architecture == kAdaptiveArchitectureId || model.m_adaptiveFeature8;
        const uint32_t expectedBottleneckLayers = model.m_adaptiveParallel32x4 ? kParallel32x4BottleneckLayers : kBottleneckLayers;
        if ((!model.m_adaptiveFilter && architecture != kArchitectureId) ||
            header->logicalInputChannels != (model.m_adaptivePairMoE ? 1u : (model.m_adaptiveFeature8 ? 8u : (model.m_adaptiveFilter ? 4u : kLogicalInputChannels))) ||
            header->packedInputChannels != model.PackedInputChannels() ||
            header->channels540 != model.Channels540() || header->channels270 != kChannels270 ||
            header->bottleneckLayers != expectedBottleneckLayers || header->tailChannels != model.TailChannels() ||
            header->scale != kUpscaleFactor || header->fp8Format != kFp8E4M3Fn ||
            header->activationLayout != kActivationLayout || header->weightLayout != kWeightLayout ||
            header->scaleConvention != kScaleConvention || header->alignment != kTensorAlignment ||
            header->descriptorSize != kDescriptorSize ||
            header->tensorCount < (kExpectedTensorCount + (model.m_adaptiveFilter ? 9u : 0u) + (model.m_adaptiveOc25 ? 1u : 0u) + (model.m_adaptivePairMoE ? 1u : 0u) + (model.m_adaptiveLearnedStencil ? 1u : 0u) + (model.m_adaptiveLearnedRgb4Dir ? 1u : 0u) + (model.m_adaptiveParallel32x4 ? 12u : 0u)) ||
            header->fileSize != model.m_storage.size() ||
            header->tocOffset != kHeaderSize || header->tocBytes != uint64_t(header->tensorCount) * kDescriptorSize ||
            header->payloadOffset < Align(header->tocOffset + header->tocBytes) ||
            header->payloadOffset > header->fileSize)
            throw std::runtime_error("unsupported or inconsistent SPYR8 header");
        if (std::any_of(
                std::begin(header->reserved),
                std::end(header->reserved),
                [](uint8_t value) { return value != 0; }))
            throw std::runtime_error("nonzero SPYR8 v1 header extension bytes");
        const auto digest = Sha256WithZeroedChecksum(model.m_storage);
        if (!std::equal(digest.begin(), digest.end(), header->checksum))
            throw std::runtime_error("SPYR8 artifact checksum mismatch: expected " +
                Hex(header->checksum, kChecksumSize) + ", observed " + Hex(digest.data(), digest.size()));
        model.m_checksumHex = Hex(digest.data(), digest.size());
        model.m_edgePhaseMaxShift = header->edgePhaseMaxShift;
        model.m_edgeGateGain = header->edgeGateGain;
        if (!std::isfinite(model.m_edgePhaseMaxShift) || model.m_edgePhaseMaxShift < 0.0f ||
            !std::isfinite(model.m_edgeGateGain) || model.m_edgeGateGain < 0.0f)
            throw std::runtime_error("invalid SPYR8 edge-phase parameters");

        std::unordered_map<std::string, TensorView> tensors;
        std::vector<std::tuple<uint64_t, uint64_t, std::string>> ranges;
        for (uint32_t index = 0; index < header->tensorCount; ++index)
        {
            const uint64_t descriptorOffset = header->tocOffset + uint64_t(index) * kDescriptorSize;
            if (descriptorOffset + sizeof(TensorDescriptor) > model.m_storage.size())
                throw std::runtime_error("SPYR8 tensor descriptor out of bounds");
            const auto* descriptor = reinterpret_cast<const TensorDescriptor*>(
                model.m_storage.data() + descriptorOffset);
            const std::string name = FixedString(descriptor->name, sizeof(descriptor->name), "tensor name");
            if (name.empty() || tensors.find(name) != tensors.end() || descriptor->rank == 0 || descriptor->rank > 4)
                throw std::runtime_error("invalid or duplicate SPYR8 tensor descriptor");
            TensorView view;
            view.dataType = static_cast<TensorDataType>(descriptor->dataType);
            view.layout = static_cast<TensorLayout>(descriptor->layout);
            view.rank = descriptor->rank;
            uint64_t elements = 1;
            for (uint32_t d = 0; d < 4; ++d)
            {
                view.shape[d] = descriptor->shape[d];
                if (d < view.rank)
                {
                    if (view.shape[d] == 0 || elements > std::numeric_limits<uint64_t>::max() / view.shape[d])
                        throw std::runtime_error("invalid SPYR8 tensor dimensions: " + name);
                    elements *= view.shape[d];
                }
                else if (view.shape[d] != 0)
                    throw std::runtime_error("nonzero unused SPYR8 dimension: " + name);
            }
            const uint64_t expectedBytes = (view.dataType == TensorDataType::NvFp4E2M1)
                ? ((elements + 1) / 2)
                : (elements * ElementSize(view.dataType));
            if (descriptor->offset % kTensorAlignment != 0 || descriptor->offset < header->payloadOffset ||
                descriptor->byteCount == 0 || descriptor->offset > header->fileSize ||
                descriptor->byteCount > header->fileSize - descriptor->offset ||
                elements > std::numeric_limits<uint64_t>::max() / ElementSize(view.dataType) ||
                descriptor->byteCount != expectedBytes ||
                std::any_of(std::begin(descriptor->reserved), std::end(descriptor->reserved), [](uint8_t v) { return v != 0; }))
                throw std::runtime_error("SPYR8 tensor bounds or byte count mismatch: " + name);
            view.data = model.m_storage.data() + descriptor->offset;
            view.byteCount = descriptor->byteCount;
            tensors.emplace(name, view);
            ranges.emplace_back(descriptor->offset, descriptor->offset + descriptor->byteCount, name);
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t index = 1; index < ranges.size(); ++index)
            if (std::get<1>(ranges[index - 1]) > std::get<0>(ranges[index]))
                throw std::runtime_error("overlapping SPYR8 tensor payloads");

        const uint32_t expectedActScales = model.m_adaptiveParallel32x4 ? kParallel32x4ActivationScaleCount : 8u;
        RequireTensor(tensors, "activation_scales", TensorDataType::Fp32,
                      TensorLayout::Vector, {expectedActScales}, model.m_activationScales);
        if (model.m_adaptiveFilter)
        {
            const uint32_t expertCount = model.m_adaptivePairMoE ? 32u : 16u;
            RequireTensor(tensors, "adaptive_basis", TensorDataType::Fp32,
                          TensorLayout::AdaptiveBasis, {25, expertCount, 4}, model.m_adaptiveBasis);
            if (model.m_adaptiveOc25)
            {
                RequireTensor(tensors, "adaptive_base", TensorDataType::Fp32,
                              TensorLayout::AdaptiveBasis, {25,4}, model.m_adaptiveBase);
                const auto* base = reinterpret_cast<const float*>(model.m_adaptiveBase.data);
                for (size_t i=0;i<100;++i)
                    if (!std::isfinite(base[i])) throw std::runtime_error("Nonfinite adaptive learned base");
            }
            if (model.m_adaptivePairMoE)
            {
                RequireTensor(tensors, "route_threshold", TensorDataType::Fp32,
                              TensorLayout::Vector, {64}, model.m_routeThreshold);
                const auto* thresh = reinterpret_cast<const float*>(model.m_routeThreshold.data);
                for (size_t i=0;i<64;++i)
                    if (!std::isfinite(thresh[i])) throw std::runtime_error("Nonfinite adaptive route threshold");
            }
            if (model.m_adaptiveLearnedStencil)
            {
                TensorView stencilView;
                RequireTensor(tensors, "analysis_stencil", TensorDataType::Fp32,
                              TensorLayout::Vector, {11}, stencilView);
                std::memcpy(&model.m_analysisStencil, stencilView.data, sizeof(LearnedAnalysisStencil));
                const auto* s = reinterpret_cast<const float*>(stencilView.data);
                for (size_t i = 0; i < 11; ++i)
                    if (!std::isfinite(s[i])) throw std::runtime_error("Nonfinite learned analysis stencil");
            }
            if (model.m_adaptiveLearnedRgb4Dir)
            {
                TensorView stencilView;
                RequireTensor(tensors, "analysis_stencil", TensorDataType::Fp32,
                              TensorLayout::Vector, {24}, stencilView);
                std::memcpy(&model.m_analysis4Dir, stencilView.data, sizeof(LearnedAnalysis4Dir));
                const auto* s = reinterpret_cast<const float*>(stencilView.data);
                for (size_t i = 0; i < 24; ++i)
                    if (!std::isfinite(s[i])) throw std::runtime_error("Nonfinite learned analysis 4-dir parameters");
            }
            TensorView guards;
            RequireTensor(tensors, "precision_guards", TensorDataType::Fp32,
                          TensorLayout::Vector, {2}, guards);
            const auto* values = reinterpret_cast<const float*>(guards.data);
            for (size_t i=0;i<2;++i)
                if (values[i]!=0.0f && values[i]!=1.0f)
                    throw std::runtime_error("Invalid adaptive precision policy");
            model.m_halfProject=values[0]==1.0f;
            model.m_halfHead=values[1]==1.0f;
            const auto* basis = reinterpret_cast<const float*>(model.m_adaptiveBasis.data);
            const size_t totalPhases = expertCount * 4;
            for (size_t kp=0;kp<totalPhases;++kp)
            {
                float dc=0.0f,l1=0.0f;
                for (size_t tap=0;tap<25;++tap)
                {
                    const float value=basis[tap*totalPhases+kp];
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite adaptive basis");
                    dc+=value;
                    l1+=std::abs(value);
                }
                if (std::abs(dc)>1.0e-5f || l1>0.5001f)
                    throw std::runtime_error("Adaptive basis violates zero-DC/L1 contract");
            }
        }
        const uint32_t channels = model.Channels540();
        const uint32_t refined = model.RefineChannels();
        if (model.m_adaptiveParallel32x4)
        {
            model.m_layers.resize(9);
            RequireWeightPair(tensors, "project", {channels,model.PackedInputChannels(),1,1}, model.m_layers[0]);
            RequireWeightPair(tensors, "compress", {64,channels*4,1,1}, model.m_layers[1]);
            RequireWeightPair(tensors, "bottleneck0", {64,32,3,3}, model.m_layers[2]);
            RequireWeightPair(tensors, "bottleneck1", {64,32,3,3}, model.m_layers[3]);
            RequireWeightPair(tensors, "bottleneck2", {64,32,3,3}, model.m_layers[4]);
            RequireWeightPair(tensors, "bottleneck3", {64,32,3,3}, model.m_layers[5]);
            RequireWeightPair(tensors, "expand", {channels*4,64,1,1}, model.m_layers[6]);
            RequireWeightPair(tensors, "refine", {refined,channels,3,3}, model.m_layers[7]);
            RequireWeightPair(tensors, "tail", {model.TailChannels(),refined,1,1}, model.m_layers[8]);
            for (size_t index=0;index<model.m_layers.size();++index)
            {
                auto& layer=model.m_layers[index];
                RequireTensor(tensors, std::string(kParallel32x4LayerNames[index])+".bias_fp8", TensorDataType::Fp32,
                    TensorLayout::Vector, {layer.bias.shape[0]}, layer.biasFp8);
            }
        }
        else
        {
            model.m_layers.resize(7);
            RequireWeightPair(tensors, "project", {channels,model.PackedInputChannels(),1,1}, model.m_layers[0]);
            RequireWeightPair(tensors, "compress", {64,channels*4,1,1}, model.m_layers[1]);
            RequireWeightPair(tensors, "bottleneck0", {64,64,3,3}, model.m_layers[2]);
            RequireWeightPair(tensors, "bottleneck1", {64,64,3,3}, model.m_layers[3]);
            RequireWeightPair(tensors, "expand", {channels*4,64,1,1}, model.m_layers[4]);
            RequireWeightPair(tensors, "refine", {refined,channels,3,3}, model.m_layers[5]);
            // The tail has no activation in the PyTorch graph.  Its PReLU vector
            // is still accepted as an all-ones identity to keep the descriptor
            // contract uniform across exporters.
            RequireWeightPair(tensors, "tail", {model.TailChannels(),refined,1,1}, model.m_layers[6]);
            constexpr const char* names[]={"project","compress","bottleneck0","bottleneck1","expand","refine","tail"};
            for (size_t index=0;index<model.m_layers.size();++index)
            {
                auto& layer=model.m_layers[index];
                if (model.m_adaptiveFilter)
                    RequireTensor(tensors, std::string(names[index])+".bias_fp8", TensorDataType::Fp32,
                        TensorLayout::Vector, {layer.bias.shape[0]}, layer.biasFp8);
                else
                    layer.biasFp8=layer.bias;
            }
        }
        return model;
    }
}
