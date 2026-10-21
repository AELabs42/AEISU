#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// SIMT only: paired output channels use HFMA2, with FP32 accumulation
// between short, 16-term half-precision partial sums. No matrix intrinsics.
template<int CI, int CO, int KS, int GROUPS, int PX>
__global__ void simt_conv(const half* __restrict__ x, const half2* __restrict__ w,
                          const float* __restrict__ bias, half2* __restrict__ out,
                          int h, int width, bool activate) {
    constexpr int CG = CI / GROUPS;
    constexpr int OP = CO / 2;
    __shared__ half tile[KS * KS * PX * CI];
    const int lane = threadIdx.x;
    const int pixel = threadIdx.y;
    const int tid = pixel * OP + lane;
    const int pos = blockIdx.x * PX + pixel;
    const int group = (lane * 2) / (CO / GROUPS);
    for (int i = tid; i < KS * KS * PX * CI; i += OP * PX) {
        const int c = i % CI;
        const int p = (i / CI) % PX;
        const int tap = i / (CI * PX);
        const int site = blockIdx.x * PX + p;
        const int yy = site / width + tap / KS - KS / 2;
        const int xx = site % width + tap % KS - KS / 2;
        tile[i] = site < h * width && yy >= 0 && yy < h && xx >= 0 && xx < width
            ? x[(yy * width + xx) * CI + c] : __float2half(0.f);
    }
    __syncthreads();
    if (pos >= h * width) return;
    float2 sum = make_float2(bias[2 * lane], bias[2 * lane + 1]);
    #pragma unroll
    for (int tap = 0; tap < KS * KS; ++tap) {
        #pragma unroll
        for (int chunk = 0; chunk < CG; chunk += 16) {
            half2 partial = __float2half2_rn(0.f);
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int c = chunk + j;
                half a = tile[(tap * PX + pixel) * CI + group * CG + c];
                partial = __hfma2(__halves2half2(a, a), w[(tap * CG + c) * OP + lane], partial);
            }
            float2 v = __half22float2(partial);
            sum.x += v.x;
            sum.y += v.y;
        }
    }
    if (activate) {
        sum.x = sum.x >= 0.f ? sum.x : sum.x * .1f;
        sum.y = sum.y >= 0.f ? sum.y : sum.y * .1f;
    }
    out[pos * OP + lane] = __floats2half2_rn(sum.x, sum.y);
}

// Winograd F(2x2,3x3): 16 transformed products instead of 36 direct
// products per input/output-channel pair and output tile. Still HFMA2 SIMT.
template<int CI, int CO, int GROUPS>
__global__ void winograd_conv(const half* __restrict__ x, const half2* __restrict__ weights,
                              const float* __restrict__ bias, half2* __restrict__ out, int h, int width) {
    constexpr int TILES = 4, OP = CO/2, CG = CI/GROUPS;
    __shared__ half transformed[16*TILES*CI];
    const int lane=threadIdx.x, tile=threadIdx.y, tid=tile*OP+lane;
    const int tw=(width+1)/2, th=(h+1)/2;
    for (int i=tid; i<TILES*CI; i+=OP*TILES) {
        const int c=i%CI, t=i/CI, site=blockIdx.x*TILES+t;
        const int oy=site/tw*2-1, ox=site%tw*2-1;
        float d[16], b[16], v[16];
        #pragma unroll
        for(int yy=0;yy<4;yy++) {
            #pragma unroll
            for(int xx=0;xx<4;xx++) {
                const int sy=oy+yy,sx=ox+xx;
                d[yy*4+xx]=site<th*tw && sy>=0 && sy<h && sx>=0 && sx<width
                    ? __half2float(x[(sy*width+sx)*CI+c]):0.f;
            }
        }
        #pragma unroll
        for(int col=0;col<4;col++) {
            b[col]=d[col]-d[8+col];
            b[4+col]=d[4+col]+d[8+col];
            b[8+col]=d[8+col]-d[4+col];
            b[12+col]=d[4+col]-d[12+col];
        }
        #pragma unroll
        for(int row=0;row<4;row++) {
            v[row*4]=b[row*4]-b[row*4+2];
            v[row*4+1]=b[row*4+1]+b[row*4+2];
            v[row*4+2]=b[row*4+2]-b[row*4+1];
            v[row*4+3]=b[row*4+1]-b[row*4+3];
        }
        #pragma unroll
        for(int j=0;j<16;j++) transformed[(j*TILES+t)*CI+c]=__float2half(v[j]);
    }
    __syncthreads();
    const int site=blockIdx.x*TILES+tile;
    if(site>=th*tw) return;
    const int group=lane*2/(CO/GROUPS);
    float2 m[16];
    #pragma unroll
    for(int j=0;j<16;j++) {
        float2 sum=make_float2(0.f,0.f);
        #pragma unroll
        for(int chunk=0;chunk<CG;chunk+=16) {
            half2 acc=__float2half2_rn(0.f);
            #pragma unroll
            for(int k=0;k<16;k++) {
                half v=transformed[(j*TILES+tile)*CI+group*CG+chunk+k];
                acc=__hfma2(__halves2half2(v,v),weights[(j*CG+chunk+k)*OP+lane],acc);
            }
            float2 p=__half22float2(acc); sum.x+=p.x;sum.y+=p.y;
        }
        m[j]=sum;
    }
    float2 a[8];
    #pragma unroll
    for(int j=0;j<4;j++) {
        a[j]=make_float2(m[j].x+m[j+4].x+m[j+8].x,m[j].y+m[j+4].y+m[j+8].y);
        a[j+4]=make_float2(m[j+4].x-m[j+8].x-m[j+12].x,m[j+4].y-m[j+8].y-m[j+12].y);
    }
    const int yy=site/tw*2,xx=site%tw*2;
    #pragma unroll
    for(int row=0;row<2;row++) {
        #pragma unroll
        for(int col=0;col<2;col++) {
            const int offset=row*4;
            float2 v=col==0 ? make_float2(a[offset].x+a[offset+1].x+a[offset+2].x,a[offset].y+a[offset+1].y+a[offset+2].y)
                :make_float2(a[offset+1].x-a[offset+2].x-a[offset+3].x,a[offset+1].y-a[offset+2].y-a[offset+3].y);
            v.x+=bias[2*lane];v.y+=bias[2*lane+1];
            v.x=v.x>=0.f?v.x:v.x*.1f;v.y=v.y>=0.f?v.y:v.y*.1f;
            if(yy+row<h && xx+col<width) out[((yy+row)*width+xx+col)*OP+lane]=__floats2half2_rn(v.x,v.y);
        }
    }
}

torch::Tensor winograd(torch::Tensor x,torch::Tensor w,torch::Tensor bias,int co,int groups) {
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast) && x.scalar_type()==torch::kFloat16,"Winograd expects NHWC half");
    const int ci=x.size(1),h=x.size(2),width=x.size(3),tiles=((h+1)/2)*((width+1)/2);
    auto out=torch::empty({1,co,h,width},x.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    auto wp=reinterpret_cast<const half2*>(w.data_ptr<at::Half>());
    auto op=reinterpret_cast<half2*>(out.data_ptr<at::Half>());
    const auto stream=at::cuda::getCurrentCUDAStream();
    if(ci==64 && co==64 && groups==2)
        winograd_conv<64,64,2><<<(tiles+3)/4,dim3(32,4),0,stream>>>(xp,wp,bias.data_ptr<float>(),op,h,width);
    else if(ci==32 && co==64 && groups==1)
        winograd_conv<32,64,1><<<(tiles+3)/4,dim3(32,4),0,stream>>>(xp,wp,bias.data_ptr<float>(),op,h,width);
    else TORCH_CHECK(false,"Unsupported Winograd shape");
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

__device__ float luma_at(const float* src,const float* p,int h,int w,int y,int x) {
    y=max(0,min(h-1,y));x=max(0,min(w-1,x));const int i=y*w+x;
    return src[i]*p[9]+src[h*w+i]*p[10]+src[2*h*w+i]*p[11];
}

__global__ void analysis(const float* __restrict__ src,const float* __restrict__ p,
                          half* __restrict__ out,int h,int w,int ph,int pw,int py,int px) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ph*pw)return;
    const int yy=i/pw,xx=i%pw,y=yy-py,x=xx-px;
    const int sy=max(0,min(h-1,y)),sx=max(0,min(w-1,x)),s=sy*w+sx;
    const float r=src[s],g=src[h*w+s],b=src[2*h*w+s];
    const int dst=((yy/2)*(pw/2)+xx/2)*32+(yy%2)*2+xx%2;
    #pragma unroll
    for(int c=0;c<3;c++)out[dst+c*4]=__float2half(r*p[c*3]+g*p[c*3+1]+b*p[c*3+2]);
    const float yc=r*p[9]+g*p[10]+b*p[11];
    const float ul=luma_at(src,p,h,w,y-1,x-1),uc=luma_at(src,p,h,w,y-1,x),ur=luma_at(src,p,h,w,y-1,x+1);
    const float cl=luma_at(src,p,h,w,y,x-1),cr=luma_at(src,p,h,w,y,x+1);
    const float dl=luma_at(src,p,h,w,y+1,x-1),dc=luma_at(src,p,h,w,y+1,x),dr=luma_at(src,p,h,w,y+1,x+1);
    const float hm=cr-cl,vm=dc-uc;
    out[dst+12]=__float2half(yc);
    out[dst+16]=__float2half((ur-ul)*p[12]+hm*p[13]+(dr-dl)*p[14]);
    out[dst+20]=__float2half((dl-ul)*p[15]+vm*p[16]+(dr-ur)*p[17]);
    out[dst+24]=__float2half((dr-ul)*p[18]+vm*p[19]+hm*p[20]);
    out[dst+28]=__float2half((ur-dl)*p[21]-vm*p[22]+hm*p[23]);
}

torch::Tensor analyze(torch::Tensor src,torch::Tensor p,int ph,int pw,int py,int px) {
    TORCH_CHECK(src.is_contiguous() && src.scalar_type()==torch::kFloat32 && src.size(0)==1,"Analysis expects NCHW FP32");
    auto out=torch::empty({1,32,ph/2,pw/2},src.options().dtype(torch::kFloat16).memory_format(at::MemoryFormat::ChannelsLast));
    const auto stream=at::cuda::getCurrentCUDAStream();
    analysis<<<(ph*pw+255)/256,256,0,stream>>>(src.data_ptr<float>(),p.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr<at::Half>()),src.size(2),src.size(3),ph,pw,py,px);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

template<int MODE>
__global__ void reconstruct(const float* __restrict__ src, const half* __restrict__ meta,
                             const float* __restrict__ base, const float* __restrict__ bank,
                             float* __restrict__ output, int sh, int sw,
                             int py, int px, int oh, int ow, int mw, bool original_source) {
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = oh / 2, width = ow / 2;
    if (site >= h * width) return;
    const int y = site / width + py, x = site % width + px;
    const int m = ((y / 2) * mw + x / 2) * (MODE == 0 ? 64 : 48);
    float a[16];
    if constexpr (MODE == 0) {
        const int parity = ((y & 1) * 2 + (x & 1)) * 16;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float v = __half2float(meta[m + parity + k]);
            const float mag = fabsf(v);
            a[k] = copysignf(mag <= 2.f ? mag : fminf(2.5f, 2.f + .05f * (mag - 2.f)), v);
        }
    }
    #pragma unroll
    for (int phase = 0; phase < 4; ++phase) {
        float r = 0.f, g = 0.f, b = 0.f;
        if constexpr (MODE != 2) {
            #pragma unroll
            for (int tap = 0; tap < 25; ++tap) {
                float filter = base[tap * 4 + phase];
                if constexpr (MODE == 0) {
                    #pragma unroll
                    for (int k = 0; k < 16; ++k)
                        filter = fmaf(a[k], bank[(tap * 16 + k) * 4 + phase], filter);
                }
                const int yy = original_source ? max(0,min(sh-1,y-py+tap/5-2)) : y+tap/5-2;
                const int xx = original_source ? max(0,min(sw-1,x-px+tap%5-2)) : x+tap%5-2;
                const int p = yy * sw + xx;
                r = fmaf(filter, src[p], r);
                g = fmaf(filter, src[sh * sw + p], g);
                b = fmaf(filter, src[2 * sh * sw + p], b);
            }
        }
        if constexpr (MODE != 0) {
            const int subpixel = ((y & 1) * 2 + phase / 2) * 4 + (x & 1) * 2 + phase % 2;
            r += __half2float(meta[m + subpixel]);
            g += __half2float(meta[m + 16 + subpixel]);
            b += __half2float(meta[m + 32 + subpixel]);
        }
        const int out = (2 * (site / width) + phase / 2) * ow + 2 * (site % width) + phase % 2;
        output[out] = fminf(1.f, fmaxf(0.f, r));
        output[oh * ow + out] = fminf(1.f, fmaxf(0.f, g));
        output[2 * oh * ow + out] = fminf(1.f, fmaxf(0.f, b));
    }
}

template<int CI, int CO, int KS, int G>
void dispatch_conv(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor out, bool act, int px) {
    const int h = x.size(2), width = x.size(3);
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    auto wp = reinterpret_cast<const half2*>(w.data_ptr<at::Half>());
    auto op = reinterpret_cast<half2*>(out.data_ptr<at::Half>());
    if (px == 16)
        simt_conv<CI,CO,KS,G,16><<<(h * width + 15) / 16, dim3(CO/2,16),0,stream>>>(xp,wp,bias.data_ptr<float>(),op,h,width,act);
    else
        simt_conv<CI,CO,KS,G,8><<<(h * width + 7) / 8, dim3(CO/2,8),0,stream>>>(xp,wp,bias.data_ptr<float>(),op,h,width,act);
}

torch::Tensor conv(torch::Tensor x, torch::Tensor w, torch::Tensor bias, int co, int ks, int groups, bool act, int px) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.size(0) == 1, "SIMT conv expects batch-one CUDA FP16");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "SIMT conv expects channels-last");
    TORCH_CHECK(px == 8 || px == 16, "Unsupported pixel tile");
    auto out = torch::empty({1,co,x.size(2),x.size(3)}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    const int ci = x.size(1);
    if (ci==32 && co==32 && ks==1 && groups==1) dispatch_conv<32,32,1,1>(x,w,bias,out,act,px);
    else if (ci==128 && co==64 && ks==1 && groups==1) dispatch_conv<128,64,1,1>(x,w,bias,out,act,px);
    else if (ci==64 && co==64 && ks==3 && groups==2) dispatch_conv<64,64,3,2>(x,w,bias,out,act,px);
    else if (ci==64 && co==128 && ks==1 && groups==1) dispatch_conv<64,128,1,1>(x,w,bias,out,act,px);
    else if (ci==32 && co==64 && ks==3 && groups==1) dispatch_conv<32,64,3,1>(x,w,bias,out,act,px);
    else if (ci==64 && co==64 && ks==1 && groups==1) dispatch_conv<64,64,1,1>(x,w,bias,out,act,px);
    else if (ci==64 && co==48 && ks==1 && groups==1) dispatch_conv<64,48,1,1>(x,w,bias,out,act,px);
    else TORCH_CHECK(false, "Unsupported SIMT convolution shape");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor head(torch::Tensor src, torch::Tensor meta, torch::Tensor base, torch::Tensor bank,
                    int py, int px, int oh, int ow, int mode, bool original_source=false) {
    TORCH_CHECK(src.is_cuda() && src.scalar_type()==torch::kFloat32 && src.is_contiguous(), "Head source must be NCHW FP32");
    TORCH_CHECK(meta.scalar_type()==torch::kFloat16 && meta.is_contiguous(at::MemoryFormat::ChannelsLast), "Head metadata must be NHWC FP16");
    TORCH_CHECK(original_source || (py>=2 && px>=2 && py+oh/2+2<=src.size(2) && px+ow/2+2<=src.size(3)), "Head crop lacks halo");
    auto out = torch::empty({1,3,oh,ow},src.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int n = (oh/2)*(ow/2);
    #define RUN(M) reconstruct<M><<<(n+127)/128,128,0,stream>>>(src.data_ptr<float>(),reinterpret_cast<const half*>(meta.data_ptr<at::Half>()),base.data_ptr<float>(),bank.data_ptr<float>(),out.data_ptr<float>(),src.size(2),src.size(3),py,px,oh,ow,meta.size(3),original_source)
    if (mode==0) { RUN(0); }
    else if (mode==1) { RUN(1); }
    else if (mode==2) { RUN(2); }
    else TORCH_CHECK(false,"Unsupported head");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

#ifndef SIDE_RUNTIME_FUSED
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("conv",&conv); m.def("head",&head); m.def("winograd",&winograd); m.def("analyze",&analyze); }
#endif
