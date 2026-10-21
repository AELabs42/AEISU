#include "SrcPyrCudnnConv.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <cudnn_frontend.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace aess::srcpyr
{
    namespace fe=cudnn_frontend;

    namespace
    {
        void CheckCuda(cudaError_t status,const char* operation)
        {
            if(status!=cudaSuccess)
                throw std::runtime_error(std::string(operation)+" failed: "+
                    cudaGetErrorString(status));
        }

        void CheckCudnn(cudnnStatus_t status,const char* operation)
        {
            if(status!=CUDNN_STATUS_SUCCESS)
                throw std::runtime_error(std::string(operation)+" failed: "+
                    cudnnGetErrorString(status));
        }

        void CheckFrontend(fe::error_t status,const char* operation)
        {
            if(status.is_bad())
                throw std::runtime_error(std::string(operation)+" failed: "+
                    status.get_message());
        }

        std::string EnvironmentValue(const char* name)
        {
            const DWORD required=GetEnvironmentVariableA(name,nullptr,0);
            if(required==0) return {};
            std::string result(required,'\0');
            const DWORD written=GetEnvironmentVariableA(
                name,result.data(),required);
            if(written==0||written>=required) return {};
            result.resize(written);
            return result;
        }

        class DeviceAllocation
        {
        public:
            ~DeviceAllocation(){Reset();}
            DeviceAllocation()=default;
            DeviceAllocation(const DeviceAllocation&)=delete;
            DeviceAllocation& operator=(const DeviceAllocation&)=delete;
            void Allocate(size_t bytes)
            {
                Reset();
                if(bytes!=0)
                {
                    CheckCuda(cudaMalloc(&m_pointer,bytes),
                        "cudaMalloc(SourcePyramid cuDNN workspace)");
                    m_bytes=bytes;
                }
            }
            void Reset() noexcept
            {
                if(m_pointer!=nullptr) cudaFree(m_pointer);
                m_pointer=nullptr;
                m_bytes=0;
            }
            void* Get() const noexcept{return m_pointer;}
            size_t Bytes() const noexcept{return m_bytes;}
        private:
            void* m_pointer=nullptr;
            size_t m_bytes=0;
        };
    }

    struct CudnnConvRuntime::Impl
    {
        static constexpr uint32_t kCacheSchema=6;
        static constexpr uintmax_t kMaximumCacheBytes=16u*1024u*1024u;

        ~Impl()
        {
            if(handle!=nullptr) cudnnDestroy(handle);
        }

        std::shared_ptr<fe::graph::Tensor_attributes> Scalar(
            const char* name,int64_t uid)
        {
            return graph->tensor(fe::graph::Tensor_attributes()
                .set_name(name).set_dim({1,1,1,1}).set_stride({1,1,1,1})
                .set_data_type(fe::DataType_t::FLOAT).set_uid(uid));
        }

        std::shared_ptr<fe::graph::Tensor_attributes> Channel(
            const char* name,int64_t uid)
        {
            const int64_t c=geometry.outputChannels;
            return graph->tensor(fe::graph::Tensor_attributes()
                .set_name(name).set_dim({1,c,1,1})
                .set_stride({c,1,c,c})
                .set_data_type(fe::DataType_t::FLOAT).set_uid(uid));
        }

        void DefineGraph()
        {
            const int64_t h=geometry.height,w=geometry.width;
            const int64_t ci=geometry.inputChannels;
            const int64_t co=geometry.outputChannels;
            const int64_t r=geometry.filter,s=geometry.filter;
            const bool fp8=geometry.dataType==CudnnConvDataType::Fp8E4m3;
            const fe::DataType_t ioType=fp8
                ? fe::DataType_t::FP8_E4M3 : fe::DataType_t::HALF;
            graph=std::make_shared<fe::graph::Graph>();
            graph->set_io_data_type(ioType)
                .set_intermediate_data_type(fe::DataType_t::FLOAT)
                .set_compute_data_type(fe::DataType_t::FLOAT);
            input=graph->tensor(fe::graph::Tensor_attributes()
                .set_name("input").set_dim({1,ci,h,w})
                .set_stride({ci*h*w,1,ci*w,ci})
                .set_data_type(ioType).set_uid(1001));
            weight=graph->tensor(fe::graph::Tensor_attributes()
                .set_name("weight").set_dim({co,ci,r,s})
                .set_stride({ci*r*s,1,ci*s,ci})
                .set_data_type(ioType).set_uid(1002));
            auto convolution=graph->conv_fprop(input,weight,
                fe::graph::Conv_fprop_attributes().set_name("conv")
                    .set_padding({int64_t(geometry.padding),
                        int64_t(geometry.padding)})
                    .set_stride({1,1}).set_dilation({1,1}));
            bias=Channel("bias",1006);
            std::shared_ptr<fe::graph::Tensor_attributes> convolutionValue=
                convolution;
            if(fp8)
            {
                inputScale=Scalar("input_scale",1003);
                weightScale=Channel("weight_scale",1005);
                auto descaledInput=graph->pointwise(
                    convolutionValue,inputScale,
                    fe::graph::Pointwise_attributes()
                        .set_name("input_descale")
                        .set_mode(fe::PointwiseMode_t::MUL));
                convolutionValue=graph->pointwise(
                    descaledInput,weightScale,
                    fe::graph::Pointwise_attributes()
                        .set_name("weight_descale")
                        .set_mode(fe::PointwiseMode_t::MUL));
            }
            auto biased=graph->pointwise(convolutionValue,bias,
                fe::graph::Pointwise_attributes().set_name("bias")
                    .set_mode(fe::PointwiseMode_t::ADD));
            if (geometry.epilogue == CudnnConvEpilogue::Fp16Bias)
            {
                output=biased;
                output->set_output(true).set_data_type(fe::DataType_t::HALF)
                    .set_uid(1008);
                return;
            }
            auto activated=graph->pointwise(biased,
                fe::graph::Pointwise_attributes()
                    .set_name("leaky_relu")
                    .set_mode(fe::PointwiseMode_t::RELU_FWD)
                    .set_relu_lower_clip_slope(0.1f));
            if(geometry.epilogue==CudnnConvEpilogue::Fp16Prelu)
            {
                output=activated;
                output->set_output(true).set_data_type(fe::DataType_t::HALF)
                    .set_uid(1008);
                return;
            }
            outputReciprocalScale=Scalar("output_reciprocal_scale",1004);
            auto requantized=graph->pointwise(activated,outputReciprocalScale,
                fe::graph::Pointwise_attributes().set_name("output_scale")
                    .set_mode(fe::PointwiseMode_t::MUL));
            requantized->set_data_type(fe::DataType_t::FP8_E4M3);
            output=requantized;
            output->set_output(true).set_uid(1008);
        }

        std::filesystem::path CachePath() const
        {
            std::filesystem::path directory;
            const std::string overrideDirectory=
                EnvironmentValue("AESS_SRCPYR_CUDNN_CACHE_DIR");
            const std::string localAppData=EnvironmentValue("LOCALAPPDATA");
            if(!overrideDirectory.empty()) directory=overrideDirectory;
            else if(!localAppData.empty())
                directory=std::filesystem::path(localAppData)/"AESS"/
                    "cudnn-plan-cache";
            else return {};
            int device=0,runtime=0,driver=0;
            cudaDeviceProp properties{};
            CheckCuda(cudaGetDevice(&device),"cudaGetDevice(SourcePyramid cache)");
            CheckCuda(cudaGetDeviceProperties(&properties,device),
                "cudaGetDeviceProperties(SourcePyramid cache)");
            CheckCuda(cudaRuntimeGetVersion(&runtime),
                "cudaRuntimeGetVersion(SourcePyramid cache)");
            CheckCuda(cudaDriverGetVersion(&driver),
                "cudaDriverGetVersion(SourcePyramid cache)");
            std::ostringstream name;
            name<<"srcpyr-"<<(geometry.cacheName?geometry.cacheName:"conv")
                <<"-v"<<kCacheSchema<<"-cudnn"<<cudnnGetVersion()
                <<"-frontend"<<CUDNN_FRONTEND_VERSION<<"-cuda"<<runtime
                <<"-driver"<<driver<<"-sm"<<properties.major<<properties.minor
                <<"-sms"<<properties.multiProcessorCount<<'-'
                <<geometry.width<<'x'<<geometry.height<<'-'
                <<geometry.inputChannels<<'x'<<geometry.outputChannels
                <<"-k"<<geometry.filter<<"p"<<geometry.padding
                <<"-e"<<static_cast<unsigned>(geometry.epilogue)
                <<"-d"<<static_cast<unsigned>(geometry.dataType)
                <<"-a"<<(geometry.autotune?1:0)<<".json";
            return directory/name.str();
        }

        bool TryLoad(const std::filesystem::path& path)
        {
            if(path.empty()) return false;
            std::error_code error;
            const uintmax_t bytes=std::filesystem::file_size(path,error);
            if(error||bytes==0||bytes>kMaximumCacheBytes) return false;
            std::ifstream cacheStream(path,std::ios::binary);
            if(!cacheStream) return false;
            std::string serialized(size_t(bytes),'\0');
            cacheStream.read(serialized.data(),std::streamsize(serialized.size()));
            if(!cacheStream) return false;
            auto loaded=fe::ExecutionPlanBuilder().setHandle(handle)
                .loadFromJson(serialized);
            plan=std::make_unique<fe::ExecutionPlan>(std::move(loaded));
            CheckCudnn(plan->get_status(),"cuDNN load SourcePyramid cached plan");
            cacheHit=true;
            return true;
        }

        std::string SerializePlan() const
        {
            std::vector<uint8_t> serializedGraph;
            CheckFrontend(graph->serialize(serializedGraph),
                "cuDNN serialize SourcePyramid plan");
            const nlohmann::json document=
                nlohmann::json::from_ubjson(serializedGraph);
            if(!document.contains("cudnn_backend_data")||
                !document["cudnn_backend_data"].is_string())
                throw std::runtime_error(
                    "cuDNN SourcePyramid graph omitted its execution plan");
            return document["cudnn_backend_data"].get<std::string>();
        }

        void Save(const std::filesystem::path& path,
            const std::string& serialized) const noexcept
        {
            if(path.empty()||serialized.empty()||
                serialized.size()>kMaximumCacheBytes) return;
            try
            {
                std::error_code error;
                std::filesystem::create_directories(path.parent_path(),error);
                if(error) return;
                std::filesystem::path temporary=path;
                temporary+=".tmp-"+std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count());
                {
                    std::ofstream cacheStream(temporary,
                        std::ios::binary|std::ios::trunc);
                    if(!cacheStream) return;
                    cacheStream.write(serialized.data(),
                        std::streamsize(serialized.size()));
                    if(!cacheStream)
                    {
                        cacheStream.close();
                        std::filesystem::remove(temporary,error);
                        return;
                    }
                }
                std::filesystem::rename(temporary,path,error);
                if(error) std::filesystem::remove(temporary,error);
            }
            catch(...)
            {
            }
        }

        void BuildPlan(const CudnnConvPointers& probe)
        {
            const std::filesystem::path cache=CachePath();
            DefineGraph();
            try
            {
                if(TryLoad(cache)) return;
            }
            catch(...)
            {
                std::error_code error;
                std::filesystem::remove(cache,error);
                plan.reset();
                cacheHit=false;
                DefineGraph();
            }
            CheckFrontend(graph->validate(),"cuDNN validate SourcePyramid graph");
            CheckFrontend(graph->build_operation_graph(handle),
                "cuDNN build SourcePyramid operation graph");
            CheckFrontend(graph->create_execution_plans({fe::HeurMode_t::A}),
                "cuDNN create SourcePyramid execution plans");
            CheckFrontend(graph->check_support(handle),
                "cuDNN check SourcePyramid graph support");
            CheckFrontend(graph->build_plans(
                handle,geometry.autotune
                    ? fe::BuildPlanPolicy_t::ALL
                    : fe::BuildPlanPolicy_t::HEURISTICS_CHOICE),
                "cuDNN build SourcePyramid execution plan");
            if (geometry.autotune && graph->get_execution_plan_count()>1)
            {
                const size_t inputBytes=size_t(geometry.width)*geometry.height*
                    geometry.inputChannels*
                    (geometry.dataType==CudnnConvDataType::Fp8E4m3?1u:2u);
                const size_t outputBytes=size_t(geometry.width)*geometry.height*
                    geometry.outputChannels*
                    (geometry.epilogue==CudnnConvEpilogue::Fp8PreluRequant
                        ? 1u : 2u);
                CheckCuda(cudaMemsetAsync(
                    probe.input,0,inputBytes,stream),
                    "cudaMemsetAsync(SourcePyramid autotune input)");
                CheckCuda(cudaMemsetAsync(
                    probe.output,0,outputBytes,stream),
                    "cudaMemsetAsync(SourcePyramid autotune output)");
                workspace.Allocate(size_t(graph->get_autotune_workspace_size()));
                std::unordered_map<
                    std::shared_ptr<fe::graph::Tensor_attributes>,void*> tensors{
                    {input,probe.input},
                    {weight,const_cast<void*>(probe.weight)},
                    {bias,const_cast<float*>(probe.bias)},
                    {output,probe.output},
                };
                if (geometry.dataType==CudnnConvDataType::Fp8E4m3)
                {
                    tensors.emplace(inputScale,
                        const_cast<float*>(probe.inputScale));
                    tensors.emplace(weightScale,
                        const_cast<float*>(probe.weightScale));
                }
                if (geometry.epilogue==CudnnConvEpilogue::Fp8PreluRequant)
                {
                    tensors.emplace(outputReciprocalScale,
                        const_cast<float*>(probe.outputReciprocalScale));
                }
                CheckFrontend(graph->autotune(
                    handle,tensors,workspace.Get()),
                    "cuDNN autotune SourcePyramid execution plans");
            }
            const std::string serialized=SerializePlan();
            auto loaded=fe::ExecutionPlanBuilder().setHandle(handle)
                .loadFromJson(serialized);
            plan=std::make_unique<fe::ExecutionPlan>(std::move(loaded));
            CheckCudnn(plan->get_status(),"cuDNN load SourcePyramid plan");
            Save(cache,serialized);
        }

        void Prepare(const std::vector<CudnnConvPointers>& pointers)
        {
            workspace.Allocate(size_t(plan->getWorkspaceSize()));
            variants.clear();
            variants.reserve(pointers.size());
            for(const CudnnConvPointers& value:pointers)
            {
                std::vector<int64_t> uids{1001,1002,1006,1008};
                std::vector<void*> data{
                    value.input,const_cast<void*>(value.weight),
                    const_cast<float*>(value.bias),value.output};
                if(geometry.dataType==CudnnConvDataType::Fp8E4m3)
                {
                    uids.insert(uids.begin()+2,{1003,1005});
                    data.insert(data.begin()+2,{
                        const_cast<float*>(value.inputScale),
                        const_cast<float*>(value.weightScale)});
                }
                if (geometry.epilogue == CudnnConvEpilogue::Fp8PreluRequant)
                {
                    uids.push_back(1004);
                    data.push_back(const_cast<float*>(
                        value.outputReciprocalScale));
                }
                auto built=fe::VariantPackBuilder()
                    .setWorkspacePointer(workspace.Get())
                    .setDataPointers(
                        static_cast<int64_t>(data.size()),data.data())
                    .setUids(
                        static_cast<int64_t>(uids.size()),uids.data()).build();
                auto variant=std::make_unique<fe::VariantPack>(std::move(built));
                CheckCudnn(variant->get_status(),
                    "cuDNN build SourcePyramid variant pack");
                variants.push_back(std::move(variant));
            }
            planName=plan->getTag();
        }

        CudnnConvGeometry geometry{};
        cudnnHandle_t handle=nullptr;
        cudaStream_t stream=nullptr;
        std::shared_ptr<fe::graph::Graph> graph;
        std::shared_ptr<fe::graph::Tensor_attributes> input,weight,inputScale;
        std::shared_ptr<fe::graph::Tensor_attributes> weightScale,bias;
        std::shared_ptr<fe::graph::Tensor_attributes> outputReciprocalScale;
        std::shared_ptr<fe::graph::Tensor_attributes> output;
        std::unique_ptr<fe::ExecutionPlan> plan;
        std::vector<std::unique_ptr<fe::VariantPack>> variants;
        DeviceAllocation workspace;
        std::string planName;
        bool cacheHit=false;
    };

    CudnnConvRuntime::CudnnConvRuntime():m_impl(std::make_unique<Impl>()){}
    CudnnConvRuntime::~CudnnConvRuntime()=default;

    void CudnnConvRuntime::Configure(
        const CudnnConvGeometry& geometry,void* cudaStream,
        const std::vector<CudnnConvPointers>& variants)
    {
        if(geometry.width==0||geometry.height==0||
            geometry.inputChannels==0||geometry.outputChannels==0||
            geometry.filter==0||cudaStream==nullptr||variants.empty())
            throw std::invalid_argument("invalid SourcePyramid cuDNN geometry");
        const bool fp8=geometry.dataType==CudnnConvDataType::Fp8E4m3;
        for(const CudnnConvPointers& value:variants)
            if(value.input==nullptr||value.weight==nullptr||
                value.bias==nullptr||value.output==nullptr||
                (fp8&&(value.inputScale==nullptr||value.weightScale==nullptr))||
                (geometry.epilogue==CudnnConvEpilogue::Fp8PreluRequant&&value.outputReciprocalScale==nullptr))
                throw std::invalid_argument(
                    "null SourcePyramid cuDNN tensor pointer");
        m_impl=std::make_unique<Impl>();
        m_impl->geometry=geometry;
        m_impl->stream=static_cast<cudaStream_t>(cudaStream);
        CheckCudnn(cudnnCreate(&m_impl->handle),
            "cudnnCreate(SourcePyramid)");
        CheckCudnn(cudnnSetStream(m_impl->handle,m_impl->stream),
            "cudnnSetStream(SourcePyramid)");
        m_impl->BuildPlan(variants.front());
        m_impl->Prepare(variants);
    }

    void CudnnConvRuntime::Execute(size_t variant)
    {
        if(!m_impl->plan||variant>=m_impl->variants.size())
            throw std::logic_error("SourcePyramid cuDNN runtime is not configured");
        CheckCudnn(cudnnBackendExecute(m_impl->handle,
            m_impl->plan->get_raw_desc(),
            m_impl->variants[variant]->get_raw_desc()),
            "cuDNN execute SourcePyramid convolution");
    }

    uint64_t CudnnConvRuntime::WorkspaceBytes() const noexcept
    {return uint64_t(m_impl->workspace.Bytes());}
    uint64_t CudnnConvRuntime::LibraryVersion() const noexcept
    {return uint64_t(cudnnGetVersion());}
    uint64_t CudnnConvRuntime::FrontendVersion() const noexcept
    {return uint64_t(CUDNN_FRONTEND_VERSION);}
    const std::string& CudnnConvRuntime::PlanName() const noexcept
    {return m_impl->planName;}
    bool CudnnConvRuntime::PlanCacheHit() const noexcept
    {return m_impl->cacheHit;}
}
