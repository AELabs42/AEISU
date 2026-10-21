// AESS generic D3D11 preprocessing and presentation shaders.
Texture2D<float4> SourceTex : register(t0);
SamplerState LinearSampler : register(s0);

cbuffer ToneConstants : register(b0)
{
    float SdrWhiteScale;
    float Gamma;
    float Saturation;
    float Contrast;
    float BlackPoint;
    float HighlightCompression;
    float SdrWhiteNits;
    float OutputWidth;
    float OutputHeight;
    float FsrSharpness;
    uint SourceTransfer;
    uint SourcePrimaries;
    float SourceCropLeft;
    float SourceCropTop;
    float SourceCropWidth;
    float SourceCropHeight;
    float SourceWidth;
    float SourceHeight;
    float ReservedPadding2;
    float ReservedPadding3;
};

struct VsOut
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

VsOut VsMain(uint vertexId : SV_VertexID)
{
    float2 pos[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0)
    };
    float2 uv[3] = {
        float2(0.0, 1.0),
        float2(0.0, -1.0),
        float2(2.0, 1.0)
    };

    VsOut output;
    output.pos = float4(pos[vertexId], 0.0, 1.0);
    output.uv = uv[vertexId];
    return output;
}

// All supported capture paths provide display-encoded SDR Rec.709 values.
// FP16 intermediate textures are numeric storage, not scRGB or PQ surfaces.
float3 EncodeSourceToWorking(float3 color) { return saturate(color); }
float3 DecodeSourceToSdr(float3 color) { return saturate(color); }

float4 LoadSourceClamp(int2 p)
{
    int2 cropSize = max(int2(SourceCropWidth, SourceCropHeight), int2(1, 1));
    int2 maxP = cropSize - 1;
    int2 sourceP = int2(SourceCropLeft, SourceCropTop) + clamp(p, int2(0, 0), maxP);
    return SourceTex.Load(int3(sourceP, 0));
}

float4 LoadFullSourceClamp(int2 p)
{
    int2 maxP = int2(OutputWidth, OutputHeight) - 1;
    if (maxP.x < 0 || maxP.y < 0)
    {
        maxP = int2(SourceWidth, SourceHeight) - 1;
    }
    return SourceTex.Load(int3(clamp(p, int2(0, 0), maxP), 0));
}

float Luma709(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float4 SampleSourceForPresent(VsOut input)
{
    float4 result = float4(0.0, 0.0, 0.0, 1.0);
    if (SourceCropWidth == OutputWidth && SourceCropHeight == OutputHeight &&
        SourceCropLeft == 0.0 && SourceCropTop == 0.0)
    {
        result = LoadFullSourceClamp(int2(input.pos.xy));
    }
    else
    {
        float2 sourceSize = max(float2(SourceWidth, SourceHeight), float2(1.0, 1.0));
        float2 cropOrigin = float2(SourceCropLeft, SourceCropTop);
        float2 cropSize = max(float2(SourceCropWidth, SourceCropHeight), float2(1.0, 1.0));
        float2 sourceUv = (cropOrigin + input.uv * cropSize) / sourceSize;
        result = SourceTex.SampleLevel(LinearSampler, sourceUv, 0.0);
    }
    return result;
}

float4 SampleSourceForPresentAtPixel(int2 p)
{
    float4 result = float4(0.0, 0.0, 0.0, 1.0);
    if (SourceCropWidth == OutputWidth && SourceCropHeight == OutputHeight &&
        SourceCropLeft == 0.0 && SourceCropTop == 0.0)
    {
        result = LoadFullSourceClamp(p);
    }
    else
    {
        float2 sourceSize = max(float2(SourceWidth, SourceHeight), float2(1.0, 1.0));
        float2 cropOrigin = float2(SourceCropLeft, SourceCropTop);
        float2 cropSize = max(float2(SourceCropWidth, SourceCropHeight), float2(1.0, 1.0));
        float2 uv = (float2(p) + 0.5) / max(float2(OutputWidth, OutputHeight), float2(1.0, 1.0));
        float2 sourceUv = (cropOrigin + uv * cropSize) / sourceSize;
        result = SourceTex.SampleLevel(LinearSampler, sourceUv, 0.0);
    }
    return result;
}

float MagicKernel(float x)
{
    x = abs(x);
    if (x < 0.5)
    {
        return 0.75 - x * x;
    }
    if (x < 1.5)
    {
        float t = 1.5 - x;
        return 0.5 * t * t;
    }
    return 0.0;
}

float4 MagicResample(uint2 outPx)
{
    float2 srcSize = max(float2(SourceCropWidth, SourceCropHeight), float2(1.0, 1.0));
    float2 dstSize = max(float2(OutputWidth, OutputHeight), float2(1.0, 1.0));
    float2 scale = srcSize / dstSize;
    float2 srcPos = (float2(outPx) + 0.5) * scale - 0.5;
    int2 basePx = int2(floor(srcPos));

    // Preserve the complete scaled B-spline footprint. The former fixed 7x7
    // loop truncated the kernel as soon as either reduction exceeded 2x,
    // causing aliasing and width-dependent loss of detail.
    const int maxRadius = 16;
    float2 filterScale = max(scale, float2(1.0, 1.0));
    int2 radius = min(int2(maxRadius, maxRadius), int2(ceil(1.5 * filterScale)));

    float4 accum = 0.0;
    float weightSum = 0.0;
    [loop]
    for (int y = -radius.y; y <= radius.y; ++y)
    {
        [loop]
        for (int x = -radius.x; x <= radius.x; ++x)
        {
            int2 samplePx = basePx + int2(x, y);
            float2 d = (float2(samplePx) - srcPos) / filterScale;
            float weight = MagicKernel(d.x) * MagicKernel(d.y);
            accum += LoadSourceClamp(samplePx) * weight;
            weightSum += weight;
        }
    }
    return accum / max(weightSum, 1.0e-5);
}

float4 MagicSharpSource(uint2 outPx)
{
    static const float weights[3] = { -0.25, 1.5, -0.25 };
    float4 accum = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            accum += LoadFullSourceClamp(int2(outPx) + int2(x, y)) * weights[x + 1] * weights[y + 1];
        }
    }
    accum.a = LoadFullSourceClamp(int2(outPx)).a;
    return accum;
}

float4 PsMain(VsOut input) : SV_Target
{
    float2 sourceSize = max(float2(SourceWidth, SourceHeight), float2(1.0, 1.0));
    float2 cropOrigin = float2(SourceCropLeft, SourceCropTop);
    float2 cropSize = max(float2(SourceCropWidth, SourceCropHeight), float2(1.0, 1.0));
    float2 sourceUv = (cropOrigin + input.uv * cropSize) / sourceSize;
    float4 color = SourceTex.SampleLevel(LinearSampler, sourceUv, 0.0);
    return float4(EncodeSourceToWorking(color.rgb), 1.0);
}

float4 PsWorkingInputMagicMain(VsOut input) : SV_Target
{
    float4 color = MagicResample(uint2(input.pos.xy));
    return float4(EncodeSourceToWorking(color.rgb), 1.0);
}

float4 PsMagicSharpEncodedMain(VsOut input) : SV_Target
{
    return float4(saturate(MagicSharpSource(uint2(input.pos.xy)).rgb), 1.0);
}

float4 PsCopySameSizeMain(VsOut input) : SV_Target
{
    return float4(saturate(LoadFullSourceClamp(int2(input.pos.xy)).rgb), 1.0);
}

float4 PsBilinearResampleMain(VsOut input) : SV_Target
{
    float2 sourceSize = max(float2(SourceWidth, SourceHeight), float2(1.0, 1.0));
    float2 cropOrigin = float2(SourceCropLeft, SourceCropTop);
    float2 cropSize = max(float2(SourceCropWidth, SourceCropHeight), float2(1.0, 1.0));
    float2 sourceUv = (cropOrigin + input.uv * cropSize) / sourceSize;
    return float4(saturate(SourceTex.SampleLevel(LinearSampler, sourceUv, 0.0).rgb), 1.0);
}



float4 PsPresentSdrMain(VsOut input) : SV_Target
{
    float4 color = MagicSharpSource(uint2(input.pos.xy));
    float3 output = saturate(color.rgb);
    return float4(output, 1.0);
}



float4 PsPresentDirectSdrMain(VsOut input) : SV_Target
{
    float4 color = SampleSourceForPresent(input);
    return float4(DecodeSourceToSdr(color.rgb), 1.0);
}











float4 PsMagicResampleMain(VsOut input) : SV_Target
{
    return float4(saturate(MagicResample(uint2(input.pos.xy)).rgb), 1.0);
}

RWTexture2D<float4> CsOutputUav : register(u0);

[numthreads(16, 16, 1)]
void CsPresentDirectSdrMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)OutputWidth || id.y >= (uint)OutputHeight)
        return;
    int2 p = int2(id.xy);
    float4 color = SampleSourceForPresentAtPixel(p);
    CsOutputUav[p] = float4(DecodeSourceToSdr(color.rgb), 1.0);
}

[numthreads(16, 16, 1)]
void CsPrepareWorkingInput(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)OutputWidth || id.y >= (uint)OutputHeight)
        return;
    int2 p = int2(id.xy);
    if (SourceCropLeft == 0.0 && SourceCropTop == 0.0 &&
        SourceCropWidth == SourceWidth && SourceCropHeight == SourceHeight &&
        SourceWidth == OutputWidth && SourceHeight == OutputHeight)
    {
        float4 color = LoadFullSourceClamp(p);
        CsOutputUav[p] = float4(EncodeSourceToWorking(color.rgb), 1.0);
        return;
    }
    float2 sourceSize = max(float2(SourceWidth, SourceHeight), float2(1.0, 1.0));
    float2 cropOrigin = float2(SourceCropLeft, SourceCropTop);
    float2 cropSize = max(float2(SourceCropWidth, SourceCropHeight), float2(1.0, 1.0));
    float2 uv = (float2(p) + 0.5) / max(float2(OutputWidth, OutputHeight), float2(1.0, 1.0));
    float2 sourceUv = (cropOrigin + uv * cropSize) / sourceSize;
    float4 color = SourceTex.SampleLevel(LinearSampler, sourceUv, 0.0);
    CsOutputUav[p] = float4(EncodeSourceToWorking(color.rgb), 1.0);
}
