#include "SrcPyrCudaRuntime.h"
#include "../NeuralInterop.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <DirectXPackedVector.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace
{
    struct PixelHalf { uint16_t r,g,b,a; };
    std::filesystem::path modelPath="models/adaptive32/adaptive_fp8.bin";
    std::filesystem::path dumpDirectory="models/adaptive32/runtime_test";

    void Check(HRESULT value,const char* operation)
    {
        if (FAILED(value))
            throw std::runtime_error(std::string(operation)+" failed");
    }

    // The runtime's CUDA/D3D11 interop is device-local.  D3D11CreateDevice
    // with a null adapter is allowed to choose the Windows primary adapter,
    // which can be an iGPU on dual-GPU systems even though CUDA selects the
    // discrete NVIDIA device.  That mismatch can hang the display driver
    // instead of returning a useful interop error.  Pin the test harness to
    // the NVIDIA adapter (matching the production capture path) before any
    // shared resources are created.
    ComPtr<IDXGIAdapter1> FindNvidiaAdapter()
    {
        ComPtr<IDXGIFactory1> factory;
        Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),
            "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> best;
        SIZE_T bestMemory = 0;
        for (UINT index = 0;; ++index)
        {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND)
                break;
            DXGI_ADAPTER_DESC1 description{};
            Check(adapter->GetDesc1(&description), "IDXGIAdapter1::GetDesc1");
            if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
                description.VendorId != 0x10DEu)
                continue;
            if (!best || description.DedicatedVideoMemory > bestMemory)
            {
                best = adapter;
                bestMemory = description.DedicatedVideoMemory;
            }
        }
        if (!best)
            throw std::runtime_error("No NVIDIA DXGI adapter was found");
        return best;
    }

    std::vector<PixelHalf> ReadTexture(
        ID3D11Device* device,ID3D11DeviceContext* context,
        ID3D11Texture2D* texture)
    {
        D3D11_TEXTURE2D_DESC description{};
        texture->GetDesc(&description);
        description.Usage=D3D11_USAGE_STAGING;
        description.BindFlags=0;
        description.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        description.MiscFlags=0;
        ComPtr<ID3D11Texture2D> staging;
        Check(device->CreateTexture2D(&description,nullptr,&staging),
            "Create staging texture");
        context->CopyResource(staging.Get(),texture);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),
            "Map staging texture");
        std::vector<PixelHalf> result(
            size_t(description.Width)*description.Height);
        for (uint32_t y=0;y<description.Height;++y)
            std::memcpy(
                result.data()+size_t(y)*description.Width,
                static_cast<const uint8_t*>(mapped.pData)+size_t(y)*mapped.RowPitch,
                size_t(description.Width)*sizeof(PixelHalf));
        context->Unmap(staging.Get(),0);
        return result;
    }

    std::vector<float> ToFloat(const std::vector<PixelHalf>& input)
    {
        std::vector<float> result(input.size()*4);
        for (size_t i=0;i<input.size();++i)
        {
            result[i*4+0]=DirectX::PackedVector::XMConvertHalfToFloat(input[i].r);
            result[i*4+1]=DirectX::PackedVector::XMConvertHalfToFloat(input[i].g);
            result[i*4+2]=DirectX::PackedVector::XMConvertHalfToFloat(input[i].b);
            result[i*4+3]=DirectX::PackedVector::XMConvertHalfToFloat(input[i].a);
        }
        return result;
    }

    void Write(const std::filesystem::path& path,const std::vector<float>& values)
    {
        std::ofstream output(path,std::ios::binary);
        if (!output) throw std::runtime_error("Could not create "+path.string());
        output.write(reinterpret_cast<const char*>(values.data()),
            std::streamsize(values.size()*sizeof(float)));
    }

    struct RunResult
    {
        std::vector<float> output;
        aess::srcpyr::FrameTimings timings{};
        uint64_t persistentBytes=0;
        float p95Ms=0;
        float controllerMedianMs=0;
        float controllerP95Ms=0;
        std::string report;
    };

    template <typename Accessor>
    float MedianTiming(
        const std::vector<aess::srcpyr::FrameTimings>& samples,
        Accessor accessor)
    {
        std::vector<float> values;
        values.reserve(samples.size());
        for (const auto& sample:samples) values.push_back(accessor(sample));
        const size_t middle=values.size()/2;
        std::nth_element(values.begin(),values.begin()+middle,values.end());
        if ((values.size()&1u)!=0) return values[middle];
        const float upper=values[middle];
        std::nth_element(values.begin(),values.begin()+middle-1,values.end());
        return 0.5f*(values[middle-1]+upper);
    }

    aess::srcpyr::FrameTimings MedianTimings(
        const std::vector<aess::srcpyr::FrameTimings>& samples)
    {
        if (samples.empty()) throw std::runtime_error("No timing samples");
        aess::srcpyr::FrameTimings result{};
#define AESS_MEDIAN_FIELD(field) result.field=MedianTiming(samples,[](const auto& value){return value.field;})
        AESS_MEDIAN_FIELD(totalCudaMs);
        if (samples.front().detailed)
        {
            AESS_MEDIAN_FIELD(interopMapMs);
            AESS_MEDIAN_FIELD(easuPackMs);
            for (size_t layer=0;layer<result.bodyLayerMs.size();++layer)
                result.bodyLayerMs[layer]=MedianTiming(
                    samples,[layer](const auto& value){return value.bodyLayerMs[layer];});
            AESS_MEDIAN_FIELD(bodyConvolutionMs);
            AESS_MEDIAN_FIELD(bodyEpilogueMs);
            AESS_MEDIAN_FIELD(tailConversionMs);
            AESS_MEDIAN_FIELD(tailGemmMs);
            AESS_MEDIAN_FIELD(compositionMs);
            AESS_MEDIAN_FIELD(reconstructionMs);
            AESS_MEDIAN_FIELD(interopUnmapMs);
            result.detailed=true;
        }
        else
        {
            result.detailed=false;
        }
#undef AESS_MEDIAN_FIELD
        return result;
    }

    RunResult Run(
        ID3D11Device* device,ID3D11DeviceContext* context,
        ID3D11Texture2D* input,ID3D11Texture2D* output,
        uint32_t width,uint32_t height,aess::srcpyr::RuntimePrecision precision,
        int warmup,int iterations,bool dump,
        aess::srcpyr::RuntimeTimingMode timingMode)
    {
        aess::srcpyr::CudaRuntime runtime(modelPath);
        runtime.SetTimingMode(timingMode);
        runtime.Configure(device,input,output,width,height,precision);

        D3D11_TEXTURE2D_DESC inputDescription{};
        input->GetDesc(&inputDescription);
        ComPtr<ID3D11Texture2D> siblingInput;
        Check(aess::neural::CreateInteropTexture(device,inputDescription,&siblingInput),
            "Create sibling input");
        context->CopyResource(siblingInput.Get(),input);
        D3D11_TEXTURE2D_DESC outputDescription{};
        output->GetDesc(&outputDescription);
        ComPtr<ID3D11Texture2D> siblingOutput;
        Check(aess::neural::CreateInteropTexture(device,outputDescription,&siblingOutput),
            "Create sibling output");

        auto sibling=runtime.CreateSibling();
        sibling->SetTimingMode(timingMode);
        sibling->Configure(
            device,siblingInput.Get(),siblingOutput.Get(),width,height,precision);
        if(!runtime.SharesExecutionEngineWith(*sibling))
            throw std::runtime_error("Sibling did not share its execution engine");

        // Reproduce AESS's production two-slot FIFO: slot one remains pending
        // while slot two is submitted to the shared highest-priority stream.
        // Distinct textures and fences make this an interop/scheduling gate,
        // rather than merely testing that a sibling can be constructed.
        runtime.ProcessAsync(precision);
        sibling->ProcessAsyncAfter(precision,&runtime);
        runtime.Wait();
        sibling->Wait();

        std::cerr<<runtime.StartupReport()<<"\n";
        for (int i=0;i<warmup;++i) runtime.Process(precision);
        std::vector<aess::srcpyr::FrameTimings> samples;
        samples.reserve(size_t(std::max(iterations,0)));
        for (int i=0;i<iterations;++i)
            samples.push_back(runtime.Process(precision));
        if (dump)
            runtime.WriteDebugTensors(
                dumpDirectory /
                (precision==aess::srcpyr::RuntimePrecision::Fp8?"fp8":
                 (precision==aess::srcpyr::RuntimePrecision::Mixed?"mixed":"fp16")));
        RunResult result;
        result.output=ToFloat(ReadTexture(device,context,output));
        result.timings=MedianTimings(samples);
        std::vector<float> totals;
        for (const auto& sample:samples) totals.push_back(sample.totalCudaMs);
        std::sort(totals.begin(),totals.end());
        result.p95Ms=totals[std::min(totals.size()-1,size_t(std::ceil(0.95*totals.size()))-1)];
        if (result.timings.detailed)
        {
            // Subtract intervals per frame, BEFORE aggregation. This yields
            // the event-0 to reconstruction-start interval, including input
            // feature packing and the complete metadata controller.
            result.controllerMedianMs=MedianTiming(samples,[](const auto& value)
                { return value.totalCudaMs-value.reconstructionMs; });
            std::vector<float> controller;
            controller.reserve(samples.size());
            for (const auto& sample:samples)
                controller.push_back(sample.totalCudaMs-sample.reconstructionMs);
            std::sort(controller.begin(),controller.end());
            result.controllerP95Ms=controller[std::min(controller.size()-1,size_t(std::ceil(0.95*controller.size()))-1)];
        }
        result.persistentBytes=runtime.PersistentDeviceBytes();
        result.report=runtime.StartupReport();
        return result;
    }
}

int main(int argc,char** argv)
{
    try
    {
        uint32_t width=64,height=48;
        int warmup=3,iterations=3;
        bool dump=true;
        bool mixedOnly=false,outputOnly=false;
        double minimumPsnr=35.0;
        aess::srcpyr::RuntimeTimingMode timingMode=aess::srcpyr::RuntimeTimingMode::TotalOnly;
        std::filesystem::path inputPath;
        for (int i=1;i<argc;++i)
        {
            const std::string argument=argv[i];
            if (argument=="--width"&&i+1<argc) width=uint32_t(std::stoul(argv[++i]));
            else if (argument=="--height"&&i+1<argc) height=uint32_t(std::stoul(argv[++i]));
            else if (argument=="--warmup"&&i+1<argc) warmup=std::stoi(argv[++i]);
            else if (argument=="--iterations"&&i+1<argc) iterations=std::stoi(argv[++i]);
            else if (argument=="--no-dump") dump=false;
            else if (argument=="--mixed-only") mixedOnly=true;
            else if (argument=="--output-only") { outputOnly=true; dump=false; }
            else if (argument=="--min-psnr"&&i+1<argc) minimumPsnr=std::stod(argv[++i]);
            else if (argument=="--timing-mode"&&i+1<argc)
            {
                const std::string m=argv[++i];
                if (m=="minimal"||m=="total"||m=="total-only") timingMode=aess::srcpyr::RuntimeTimingMode::TotalOnly;
                else if (m=="detailed") timingMode=aess::srcpyr::RuntimeTimingMode::Detailed;
                else throw std::runtime_error("Invalid timing mode: "+m);
            }
            else if (argument=="--model"&&i+1<argc) modelPath=argv[++i];
            else if (argument=="--dump-dir"&&i+1<argc) dumpDirectory=argv[++i];
            else if (argument=="--input"&&i+1<argc) inputPath=argv[++i];
#ifdef AESS_COEFFICIENT_CONTEXT_EXPERIMENT
            else if (argument=="--enable-context") _putenv_s("AESS_CONTEXT_ENABLE","1");
            else if (argument=="--disable-context") _putenv_s("AESS_CONTEXT_ENABLE","0");
            else if (argument=="--context-gamma"&&i+1<argc) _putenv_s("AESS_CONTEXT_GAMMA",argv[++i]);
            else if (argument=="--context-beta"&&i+1<argc) _putenv_s("AESS_CONTEXT_BETA",argv[++i]);
            else if (argument=="--context-matrix"&&i+1<argc) _putenv_s("AESS_CONTEXT_MATRIX",argv[++i]);
#endif
            else throw std::runtime_error("Unknown or incomplete test option: "+argument);
        }
        if (!width || !height || width>8192 || height>8192 || warmup<0 || iterations<1)
            throw std::runtime_error("Invalid test dimensions or iteration count");

        D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL selected{};
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        const ComPtr<IDXGIAdapter1> nvidiaAdapter=FindNvidiaAdapter();
        Check(D3D11CreateDevice(nvidiaAdapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,
            levels,UINT(std::size(levels)),D3D11_SDK_VERSION,&device,&selected,&context),
            "D3D11CreateDevice");

        std::vector<PixelHalf> source(size_t(width)*height);
        for (uint32_t y=0;y<height;++y)
            for (uint32_t x=0;x<width;++x)
            {
                const float r=float(x)/std::max(1u,width-1);
                const float g=float(y)/std::max(1u,height-1);
                const float b=float((x*17+y*29)%97)/96.0f;
                source[size_t(y)*width+x]={
                    DirectX::PackedVector::XMConvertFloatToHalf(r),
                    DirectX::PackedVector::XMConvertFloatToHalf(g),
                    DirectX::PackedVector::XMConvertFloatToHalf(b),
                    DirectX::PackedVector::XMConvertFloatToHalf(1.0f)};
            }
        if (!inputPath.empty())
        {
            std::ifstream inputFile(inputPath,std::ios::binary|std::ios::ate);
            const size_t byteCount=source.size()*4*sizeof(float);
            if (!inputFile || inputFile.tellg()!=std::streamoff(byteCount))
                throw std::runtime_error("Input must contain width*height RGBA32F pixels");
            inputFile.seekg(0);
            std::vector<float> values(source.size()*4);
            if (!inputFile.read(reinterpret_cast<char*>(values.data()),std::streamsize(byteCount)))
                throw std::runtime_error("Input read failed");
            for (size_t i=0;i<source.size();++i)
                source[i]={DirectX::PackedVector::XMConvertFloatToHalf(values[i*4]),
                    DirectX::PackedVector::XMConvertFloatToHalf(values[i*4+1]),
                    DirectX::PackedVector::XMConvertFloatToHalf(values[i*4+2]),
                    DirectX::PackedVector::XMConvertFloatToHalf(1.0f)};
        }
        D3D11_TEXTURE2D_DESC description{};
        description.Width=width;
        description.Height=height;
        description.MipLevels=1;
        description.ArraySize=1;
        description.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        description.SampleDesc.Count=1;
        description.Usage=D3D11_USAGE_DEFAULT;
        description.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_RENDER_TARGET;
        description.MiscFlags=D3D11_RESOURCE_MISC_SHARED;
        D3D11_SUBRESOURCE_DATA initial{};
        initial.pSysMem=source.data();
        initial.SysMemPitch=width*sizeof(PixelHalf);
        ComPtr<ID3D11Texture2D> input;
        Check(aess::neural::CreateInteropTexture(device.Get(),description,&input),"Create input");
        context->UpdateSubresource(input.Get(),0,nullptr,initial.pSysMem,initial.SysMemPitch,0);
        description.Width*=2;
        description.Height*=2;
        ComPtr<ID3D11Texture2D> output;
        Check(aess::neural::CreateInteropTexture(device.Get(),description,&output),"Create output");

        if (mixedOnly)
        {
            const RunResult mixed=Run(device.Get(),context.Get(),input.Get(),output.Get(),
                width,height,aess::srcpyr::RuntimePrecision::Mixed,warmup,iterations,dump,timingMode);
            for (size_t i=0;i<mixed.output.size();++i)
                if (!std::isfinite(mixed.output[i]) || ((i&3)==3 && mixed.output[i]!=1.0f))
                    throw std::runtime_error("Invalid mixed output or alpha");
            if (dump || outputOnly)
            {
                std::filesystem::create_directories(dumpDirectory);
                Write(dumpDirectory/"output_mixed_rgba_f32.bin",mixed.output);
            }
            // This mode measures deployed inference only. It intentionally
            // makes no FP16-reference PSNR assertion; use the default full
            // precision suite for that independent numerical correctness gate.
            std::cout<<"Mixed runtime PASS "<<width<<'x'<<height
                <<" total_median="<<mixed.timings.totalCudaMs<<"ms total_p95="<<mixed.p95Ms<<"ms\n";
            if (mixed.timings.detailed)
            {
                std::cout<<"Mixed front: controller_median="<<mixed.controllerMedianMs
                    <<"ms controller_p95="<<mixed.controllerP95Ms
                    <<"ms input_pack="<<mixed.timings.easuPackMs
                    <<"ms reconstruction="<<mixed.timings.reconstructionMs<<"ms\n";
                std::cout<<"Mixed stages: feature_pack="<<mixed.timings.easuPackMs
                    <<"ms conv="<<mixed.timings.bodyConvolutionMs
                    <<"ms epilogue="<<mixed.timings.bodyEpilogueMs
                    <<"ms reshape="<<mixed.timings.tailConversionMs
                    <<"ms tail="<<mixed.timings.tailGemmMs
                    <<"ms groups=["<<mixed.timings.bodyLayerMs[0]<<','
                    <<mixed.timings.bodyLayerMs[1]<<','
                    <<mixed.timings.bodyLayerMs[2]<<"]ms expand="
                    <<mixed.timings.compositionMs<<"ms refine="
                    <<(mixed.timings.bodyLayerMs[2]-mixed.timings.compositionMs)<<"ms\n";
            }
            std::cout<<mixed.report<<"\n";
            return 0;
        }

        const RunResult fp8=Run(device.Get(),context.Get(),input.Get(),output.Get(),
            width,height,aess::srcpyr::RuntimePrecision::Fp8,warmup,iterations,dump,timingMode);
        const RunResult fp16=Run(device.Get(),context.Get(),input.Get(),output.Get(),
            width,height,aess::srcpyr::RuntimePrecision::Fp16,warmup,iterations,dump,timingMode);
        const RunResult mixed=Run(device.Get(),context.Get(),input.Get(),output.Get(),
            width,height,aess::srcpyr::RuntimePrecision::Mixed,warmup,iterations,dump,timingMode);
        if (fp8.output.size()!=fp16.output.size() || mixed.output.size()!=fp16.output.size())
            throw std::runtime_error("Output size mismatch");
        double squared=0.0;
        double squaredMixed=0.0;
        size_t count=0;
        size_t invalidAlphaCount=0;
        float minimumAlpha=1.0f,maximumAlpha=1.0f;
        for (size_t i=0;i<fp8.output.size();++i)
        {
            if (!std::isfinite(fp8.output[i])||!std::isfinite(fp16.output[i])||!std::isfinite(mixed.output[i]))
                throw std::runtime_error("Non-finite output");
            if ((i&3)==3)
            {
                minimumAlpha=std::min({minimumAlpha,fp8.output[i],fp16.output[i],mixed.output[i]});
                maximumAlpha=std::max({maximumAlpha,fp8.output[i],fp16.output[i],mixed.output[i]});
                if (std::abs(fp8.output[i]-1.0f)>0.001f||
                    std::abs(fp16.output[i]-1.0f)>0.001f||
                    std::abs(mixed.output[i]-1.0f)>0.001f) ++invalidAlphaCount;
                continue;
            }
            const double difference=double(fp8.output[i])-fp16.output[i];
            squared+=difference*difference;
            const double diffMixed=double(mixed.output[i])-fp16.output[i];
            squaredMixed+=diffMixed*diffMixed;
            ++count;
        }
        const double mse=squared/std::max<size_t>(1,count);
        const double psnr=mse==0.0?INFINITY:10.0*std::log10(1.0/mse);
        const double mseMixed=squaredMixed/std::max<size_t>(1,count);
        const double psnrMixed=mseMixed==0.0?INFINITY:10.0*std::log10(1.0/mseMixed);
        std::cerr<<"FP8/FP16 MSE="<<mse<<" PSNR="<<psnr<<"dB\n";
        std::cerr<<"Mixed/FP16 MSE="<<mseMixed<<" PSNR="<<psnrMixed<<"dB\n";
        if (dump || outputOnly)
        {
            const std::filesystem::path directory=dumpDirectory;
            std::filesystem::create_directories(directory);
            Write(directory/"input_rgba_f32.bin",ToFloat(source));
            Write(directory/"output_fp8_rgba_f32.bin",fp8.output);
            Write(directory/"output_fp16_rgba_f32.bin",fp16.output);
            Write(directory/"output_mixed_rgba_f32.bin",mixed.output);
        }
        if (invalidAlphaCount!=0)
            throw std::runtime_error("Invalid alpha output: count="+
                std::to_string(invalidAlphaCount)+" range="+
                std::to_string(minimumAlpha)+".."+
                std::to_string(maximumAlpha));
        if (std::isnan(psnr) || psnr < minimumPsnr)
            throw std::runtime_error("FP8/FP16 PSNR gate failed: " + std::to_string(psnr) + " dB < " + std::to_string(minimumPsnr) + " dB");
        if (std::isnan(psnrMixed) || psnrMixed < minimumPsnr)
            throw std::runtime_error("Mixed/FP16 PSNR gate failed: " + std::to_string(psnrMixed) + " dB < " + std::to_string(minimumPsnr) + " dB");
        std::cout<<"SourcePyramid runtime PASS "<<width<<'x'<<height
            <<" FP8="<<fp8.timings.totalCudaMs<<"ms FP16="
            <<fp16.timings.totalCudaMs<<"ms Mixed="
            <<mixed.timings.totalCudaMs<<"ms PSNR="<<psnr<<"dB PSNR_Mixed="<<psnrMixed<<"dB\n"
            <<"CUDA p95: FP8="<<fp8.p95Ms<<"ms FP16="<<fp16.p95Ms<<"ms Mixed="<<mixed.p95Ms<<"ms\n";
        if (fp8.timings.detailed)
        {
            std::cout<<"FP8 stages: feature_pack="<<fp8.timings.easuPackMs
                <<"ms conv="<<fp8.timings.bodyConvolutionMs
                <<"ms epilogue="<<fp8.timings.bodyEpilogueMs
                <<"ms reshape="<<fp8.timings.tailConversionMs
                <<"ms tail="<<fp8.timings.tailGemmMs
                <<"ms groups=["<<fp8.timings.bodyLayerMs[0]<<','
                <<fp8.timings.bodyLayerMs[1]<<','
                <<fp8.timings.bodyLayerMs[2]<<"]ms expand="
                <<fp8.timings.compositionMs<<"ms refine="
                <<(fp8.timings.bodyLayerMs[2]-fp8.timings.compositionMs)<<"ms\n"
                <<"FP16 stages: feature_pack="<<fp16.timings.easuPackMs
                <<"ms conv="<<fp16.timings.bodyConvolutionMs
                <<"ms epilogue="<<fp16.timings.bodyEpilogueMs
                <<"ms reshape="<<fp16.timings.tailConversionMs
                <<"ms tail="<<fp16.timings.tailGemmMs
                <<"ms groups=["<<fp16.timings.bodyLayerMs[0]<<','
                <<fp16.timings.bodyLayerMs[1]<<','
                <<fp16.timings.bodyLayerMs[2]<<"]ms expand="
                <<fp16.timings.compositionMs<<"ms refine="
                <<(fp16.timings.bodyLayerMs[2]-fp16.timings.compositionMs)<<"ms\n"
                <<"Mixed stages: feature_pack="<<mixed.timings.easuPackMs
                <<"ms conv="<<mixed.timings.bodyConvolutionMs
                <<"ms epilogue="<<mixed.timings.bodyEpilogueMs
                <<"ms reshape="<<mixed.timings.tailConversionMs
                <<"ms tail="<<mixed.timings.tailGemmMs
                <<"ms groups=["<<mixed.timings.bodyLayerMs[0]<<','
                <<mixed.timings.bodyLayerMs[1]<<','
                <<mixed.timings.bodyLayerMs[2]<<"]ms expand="
                <<mixed.timings.compositionMs<<"ms refine="
                <<(mixed.timings.bodyLayerMs[2]-mixed.timings.compositionMs)<<"ms\n"
                <<"Reconstruction: FP8="<<fp8.timings.reconstructionMs<<"ms FP16="
                <<fp16.timings.reconstructionMs<<"ms Mixed="<<mixed.timings.reconstructionMs<<"ms\n";
        }
        std::cout<<mixed.report<<"\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr<<"SourcePyramid runtime FAIL: "<<error.what()<<"\n";
        return 1;
    }
}
