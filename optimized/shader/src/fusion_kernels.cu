#define SIDE_RUNTIME_FUSED
#include "shader_kernels.cu"

// One source-neighborhood load feeds all four phases; base/bank phases are float4.
template<int MODE>
__global__ void phase_shared_head(const float* __restrict__ src,const half* __restrict__ meta,
                                  const float4* __restrict__ base,const float4* __restrict__ bank,
                                  float* __restrict__ output,int sh,int sw,int py,int px,int mw) {
    const int site=blockIdx.x*blockDim.x+threadIdx.x;
    if(site>=sh*sw)return;
    const int y=site/sw,x=site%sw,my=y+py,mx=x+px;
    const int cell=(my/2)*mw+mx/2,parity=(my%2)*2+mx%2;
    const half* coefficients=meta+cell*(MODE==0?64:48);
    float a[16];
    if constexpr(MODE==0) {
        #pragma unroll
        for(int k=0;k<16;k++) {
            const float v=__half2float(coefficients[parity*16+k]),mag=fabsf(v);
            a[k]=copysignf(mag<=2.f?mag:fminf(2.5f,2.f+.05f*(mag-2.f)),v);
        }
    }
    float4 r=make_float4(0.f,0.f,0.f,0.f),g=r,b=r;
    if constexpr(MODE!=2) {
        #pragma unroll
        for(int tap=0;tap<25;tap++) {
            float4 filter=base[tap];
            if constexpr(MODE==0) {
                #pragma unroll
                for(int k=0;k<16;k++) {
                    const float4 w=bank[tap*16+k];
                    filter.x=fmaf(a[k],w.x,filter.x);filter.y=fmaf(a[k],w.y,filter.y);
                    filter.z=fmaf(a[k],w.z,filter.z);filter.w=fmaf(a[k],w.w,filter.w);
                }
            }
            const int sy=max(0,min(sh-1,y+tap/5-2)),sx=max(0,min(sw-1,x+tap%5-2)),s=sy*sw+sx;
            const float rv=src[s],gv=src[sh*sw+s],bv=src[2*sh*sw+s];
            r.x=fmaf(filter.x,rv,r.x);r.y=fmaf(filter.y,rv,r.y);r.z=fmaf(filter.z,rv,r.z);r.w=fmaf(filter.w,rv,r.w);
            g.x=fmaf(filter.x,gv,g.x);g.y=fmaf(filter.y,gv,g.y);g.z=fmaf(filter.z,gv,g.z);g.w=fmaf(filter.w,gv,g.w);
            b.x=fmaf(filter.x,bv,b.x);b.y=fmaf(filter.y,bv,b.y);b.z=fmaf(filter.z,bv,b.z);b.w=fmaf(filter.w,bv,b.w);
        }
    }
    #pragma unroll
    for(int phase=0;phase<4;phase++) {
        float rv=reinterpret_cast<float*>(&r)[phase],gv=reinterpret_cast<float*>(&g)[phase],bv=reinterpret_cast<float*>(&b)[phase];
        if constexpr(MODE!=0) {
            const int p=((parity/2)*2+phase/2)*4+(parity%2)*2+phase%2;
            rv+=__half2float(coefficients[p]);gv+=__half2float(coefficients[16+p]);bv+=__half2float(coefficients[32+p]);
        }
        const int oh=sh*2,ow=sw*2,dst=(y*2+phase/2)*ow+x*2+phase%2;
        output[dst]=fminf(1.f,fmaxf(0.f,rv));output[oh*ow+dst]=fminf(1.f,fmaxf(0.f,gv));
        output[2*oh*ow+dst]=fminf(1.f,fmaxf(0.f,bv));
    }
}

torch::Tensor shared_head(torch::Tensor src,torch::Tensor metadata,torch::Tensor base,torch::Tensor bank,int py,int px,int mode,int block) {
    const int sh=src.size(2),sw=src.size(3),n=sh*sw;
    auto out=torch::empty({1,3,sh*2,sw*2},src.options());
    const auto stream=at::cuda::getCurrentCUDAStream();
    #define SHARED_HEAD(M) phase_shared_head<M><<<(n+block-1)/block,block,0,stream>>>(src.data_ptr<float>(),reinterpret_cast<const half*>(metadata.data_ptr<at::Half>()),reinterpret_cast<const float4*>(base.data_ptr<float>()),reinterpret_cast<const float4*>(bank.data_ptr<float>()),out.data_ptr<float>(),sh,sw,py,px,metadata.size(3))
    if(mode==0){SHARED_HEAD(0);}else if(mode==1){SHARED_HEAD(1);}else if(mode==2){SHARED_HEAD(2);}
    else TORCH_CHECK(false,"Unsupported output head");
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Address the shuffle in the consumer/producer instead of materializing it.
template<bool EXPAND, int PX>
__global__ void fused_shuffle_conv(const half* __restrict__ x, const half* __restrict__ skip,
                                   const half2* __restrict__ weights, const float* __restrict__ bias,
                                   half2* __restrict__ out, int h, int width) {
    constexpr int CI = EXPAND ? 64 : 128, CO = EXPAND ? 128 : 64, OP = CO/2;
    __shared__ half tile[PX*CI];
    const int lane=threadIdx.x, pixel=threadIdx.y, tid=pixel*OP+lane;
    const int site=blockIdx.x*PX+pixel;
    for(int i=tid;i<PX*CI;i+=PX*OP) {
        const int p=blockIdx.x*PX+i/CI,c=i%CI;
        if(p>=h*width) tile[i]=__float2half(0.f);
        else if constexpr(EXPAND) tile[i]=x[p*CI+c];
        else {
            const int y=2*(p/width)+(c%4)/2, xx=2*(p%width)+c%2;
            tile[i]=x[(y*(width*2)+xx)*32+c/4];
        }
    }
    __syncthreads();
    if(site>=h*width)return;
    float2 sum=make_float2(bias[2*lane],bias[2*lane+1]);
    #pragma unroll
    for(int chunk=0;chunk<CI;chunk+=16) {
        half2 partial=__float2half2_rn(0.f);
        #pragma unroll
        for(int k=0;k<16;k++) {
            const half v=tile[pixel*CI+chunk+k];
            partial=__hfma2(__halves2half2(v,v),weights[(chunk+k)*OP+lane],partial);
        }
        const float2 v=__half22float2(partial);sum.x+=v.x;sum.y+=v.y;
    }
    sum.x=sum.x>=0.f?sum.x:sum.x*.1f;sum.y=sum.y>=0.f?sum.y:sum.y*.1f;
    half2 result=__floats2half2_rn(sum.x,sum.y);
    if constexpr(EXPAND) {
        const int c=2*lane, parity=c%4, channel=c/4;
        const int y=2*(site/width)+parity/2, xx=2*(site%width)+parity%2;
        const int s=(y*(width*2)+xx)*32+channel;
        const int s2=s+32;
        const half2 prior=__halves2half2(skip[s],skip[s2]);
        result=__hadd2(result,prior);
        half* dst=reinterpret_cast<half*>(out);
        dst[s]=__low2half(result);dst[s2]=__high2half(result);
    } else out[site*OP+lane]=result;
}

torch::Tensor shuffle_conv(torch::Tensor x,torch::Tensor skip,torch::Tensor weights,torch::Tensor bias,bool expand,int px) {
    const int h=expand?x.size(2):x.size(2)/2,width=expand?x.size(3):x.size(3)/2;
    auto out=torch::empty({1,expand?32:64,expand?h*2:h,expand?width*2:width},
                          x.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto stream=at::cuda::getCurrentCUDAStream();
    auto xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>()),sp=reinterpret_cast<const half*>(skip.data_ptr<at::Half>());
    auto wp=reinterpret_cast<const half2*>(weights.data_ptr<at::Half>());
    auto op=reinterpret_cast<half2*>(out.data_ptr<at::Half>());
    #define SHUFFLE(E,P) fused_shuffle_conv<E,P><<<(h*width+P-1)/P,dim3(E?64:32,P),0,stream>>>(xp,sp,wp,bias.data_ptr<float>(),op,h,width)
    if(expand) { if(px==16) {SHUFFLE(true,16);} else {SHUFFLE(true,8);} }
    else { if(px==16) {SHUFFLE(false,16);} else {SHUFFLE(false,8);} }
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Analysis and project share their 32-channel packed features in shared memory.
template<int PX,bool INT8=false>
__global__ void analysis_project(const float* __restrict__ src,const float* __restrict__ p,
                                  const half2* __restrict__ weights,const float* __restrict__ bias,
                                  half2* __restrict__ out,int h,int width,int ph,int pw,int py,int px,
                                  const int* qweights=nullptr,const float* factors=nullptr,float inv_in=0.f,float inv_out=0.f) {
    __shared__ half features[PX*32];
    const int lane=threadIdx.x,pixel=threadIdx.y,tid=pixel*16+lane;
    for(int i=tid;i<PX*4;i+=PX*16) {
        const int cell=blockIdx.x*PX+i/4,parity=i%4;
        const int y=(cell/(pw/2))*2+parity/2-py,x=(cell%(pw/2))*2+parity%2-px;
        const int sy=max(0,min(h-1,y)),sx=max(0,min(width-1,x)),s=sy*width+sx;
        const float r=src[s],g=src[h*width+s],b=src[2*h*width+s];
        const int dst=(i/4)*32+parity;
        #pragma unroll
        for(int c=0;c<3;c++)features[dst+c*4]=__float2half(r*p[c*3]+g*p[c*3+1]+b*p[c*3+2]);
        const float yc=r*p[9]+g*p[10]+b*p[11];
        const float ul=luma_at(src,p,h,width,y-1,x-1),ur=luma_at(src,p,h,width,y-1,x+1);
        const float cl=luma_at(src,p,h,width,y,x-1),cr=luma_at(src,p,h,width,y,x+1);
        const float dl=luma_at(src,p,h,width,y+1,x-1),dr=luma_at(src,p,h,width,y+1,x+1);
        const float uc=luma_at(src,p,h,width,y-1,x),dc=luma_at(src,p,h,width,y+1,x);
        const float hm=cr-cl,vm=dc-uc;
        features[dst+12]=__float2half(yc);
        features[dst+16]=__float2half((ur-ul)*p[12]+hm*p[13]+(dr-dl)*p[14]);
        features[dst+20]=__float2half((dl-ul)*p[15]+vm*p[16]+(dr-ur)*p[17]);
        features[dst+24]=__float2half((dr-ul)*p[18]+vm*p[19]+hm*p[20]);
        features[dst+28]=__float2half((ur-dl)*p[21]-vm*p[22]+hm*p[23]);
    }
    __syncthreads();
    const int site=blockIdx.x*PX+pixel;
    if(site>=(ph/2)*(pw/2))return;
    if constexpr(INT8) {
        int acc0=0,acc1=0;
        #pragma unroll
        for(int word=0;word<8;word++) {
            unsigned int bits=0;
            #pragma unroll
            for(int k=0;k<4;k++) {
                const float value=__half2float(features[pixel*32+word*4+k]);
                const int q=max(-127,min(127,__float2int_rn(value*inv_in)));
                bits|=(static_cast<unsigned int>(q)&255u)<<(k*8);
            }
            acc0=__dp4a(static_cast<int>(bits),qweights[word*32+2*lane],acc0);
            acc1=__dp4a(static_cast<int>(bits),qweights[word*32+2*lane+1],acc1);
        }
        float v0=static_cast<float>(acc0)*factors[2*lane]+bias[2*lane];
        float v1=static_cast<float>(acc1)*factors[2*lane+1]+bias[2*lane+1];
        v0=v0>=0.f?v0:v0*.1f;v1=v1>=0.f?v1:v1*.1f;
        auto dst=reinterpret_cast<signed char*>(out);
        dst[site*32+2*lane]=static_cast<signed char>(max(-127,min(127,__float2int_rn(v0*inv_out))));
        dst[site*32+2*lane+1]=static_cast<signed char>(max(-127,min(127,__float2int_rn(v1*inv_out))));
        return;
    }
    float2 sum=make_float2(bias[2*lane],bias[2*lane+1]);
    #pragma unroll
    for(int chunk=0;chunk<32;chunk+=16) {
        half2 part=__float2half2_rn(0.f);
        #pragma unroll
        for(int k=0;k<16;k++) {
            half v=features[pixel*32+chunk+k];
            part=__hfma2(__halves2half2(v,v),weights[(chunk+k)*16+lane],part);
        }
        float2 v=__half22float2(part);sum.x+=v.x;sum.y+=v.y;
    }
    sum.x=sum.x>=0.f?sum.x:sum.x*.1f;sum.y=sum.y>=0.f?sum.y:sum.y*.1f;
    out[site*16+lane]=__floats2half2_rn(sum.x,sum.y);
}

torch::Tensor project_source_int8(torch::Tensor src,torch::Tensor p,torch::Tensor w,torch::Tensor factors,torch::Tensor bias,
                                  int ph,int pw,int py,int px,double input_scale,double output_scale) {
    auto out=torch::empty({1,32,ph/2,pw/2},src.options().dtype(torch::kInt8).memory_format(at::MemoryFormat::ChannelsLast));
    const int n=(ph/2)*(pw/2);
    analysis_project<8,true><<<(n+7)/8,dim3(16,8),0,at::cuda::getCurrentCUDAStream()>>>(src.data_ptr<float>(),p.data_ptr<float>(),
        nullptr,bias.data_ptr<float>(),reinterpret_cast<half2*>(out.data_ptr<int8_t>()),src.size(2),src.size(3),ph,pw,py,px,
        w.data_ptr<int>(),factors.data_ptr<float>(),1.f/static_cast<float>(input_scale),1.f/static_cast<float>(output_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor project_source(torch::Tensor src,torch::Tensor p,torch::Tensor w,torch::Tensor bias,int ph,int pw,int py,int px,int tile) {
    auto out=torch::empty({1,32,ph/2,pw/2},src.options().dtype(torch::kFloat16).memory_format(at::MemoryFormat::ChannelsLast));
    const auto stream=at::cuda::getCurrentCUDAStream();
    const int n=(ph/2)*(pw/2);
    #define PROJECT(P) analysis_project<P><<<(n+P-1)/P,dim3(16,P),0,stream>>>(src.data_ptr<float>(),p.data_ptr<float>(),reinterpret_cast<const half2*>(w.data_ptr<at::Half>()),bias.data_ptr<float>(),reinterpret_cast<half2*>(out.data_ptr<at::Half>()),src.size(2),src.size(3),ph,pw,py,px)
    if(tile==16){PROJECT(16);}else{PROJECT(8);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

template<int MODE>
__device__ void reconstruct_local(const float* src,const half* meta,const float* base,const float* bank,float* output,
                                  int sh,int sw,int y,int x,int parity) {
    float a[16];
    if constexpr(MODE==0) {
        #pragma unroll
        for(int k=0;k<16;k++) {
            const float v=__half2float(meta[parity*16+k]),mag=fabsf(v);
            a[k]=copysignf(mag<=2.f?mag:fminf(2.5f,2.f+.05f*(mag-2.f)),v);
        }
    }
    const int oh=sh*2,ow=sw*2;
    #pragma unroll
    for(int phase=0;phase<4;phase++) {
        float r=0.f,g=0.f,b=0.f;
        if constexpr(MODE!=2) {
            #pragma unroll
            for(int tap=0;tap<25;tap++) {
                float filter=base[tap*4+phase];
                if constexpr(MODE==0) {
                    #pragma unroll
                    for(int k=0;k<16;k++)filter=fmaf(a[k],bank[(tap*16+k)*4+phase],filter);
                }
                const int yy=max(0,min(sh-1,y+tap/5-2)),xx=max(0,min(sw-1,x+tap%5-2)),s=yy*sw+xx;
                r=fmaf(filter,src[s],r);g=fmaf(filter,src[sh*sw+s],g);b=fmaf(filter,src[2*sh*sw+s],b);
            }
        }
        if constexpr(MODE!=0) {
            const int p=((parity/2)*2+phase/2)*4+(parity%2)*2+phase%2;
            r+=__half2float(meta[p]);g+=__half2float(meta[16+p]);b+=__half2float(meta[32+p]);
        }
        const int dst=(y*2+phase/2)*ow+x*2+phase%2;
        output[dst]=fminf(1.f,fmaxf(0.f,r));output[oh*ow+dst]=fminf(1.f,fmaxf(0.f,g));
        output[2*oh*ow+dst]=fminf(1.f,fmaxf(0.f,b));
    }
}

// Refine, linear tail, and optionally reconstruction share block-local tensors.
template<int CO,bool FINAL=false,int MODE=0>
__global__ void refine_tail(const half* __restrict__ x,const half2* __restrict__ weights,const float* __restrict__ bias,
                            const half2* __restrict__ tail_weights,const float* __restrict__ tail_bias,
                            half2* __restrict__ out,int h,int width,
                            const float* src=nullptr,const float* base=nullptr,const float* bank=nullptr,float* rgb=nullptr,
                            int sh=0,int sw=0,int py=0,int px=0) {
    constexpr int TILES=4,CI=32,OP=32;
    __shared__ half transformed[16*TILES*CI];
    __shared__ half refined[TILES*4*64];
    __shared__ half metadata[FINAL?TILES*4*CO:1];
    const int lane=threadIdx.x,tile=threadIdx.y,tid=tile*OP+lane;
    const int tw=(width+1)/2,th=(h+1)/2;
    for(int i=tid;i<TILES*CI;i+=OP*TILES) {
        const int c=i%CI,t=i/CI,site=blockIdx.x*TILES+t;
        const int oy=site/tw*2-1,ox=site%tw*2-1;
        float d[16],b[16],v[16];
        #pragma unroll
        for(int yy=0;yy<4;yy++) {
            #pragma unroll
            for(int xx=0;xx<4;xx++) {
                const int sy=oy+yy,sx=ox+xx;
                d[yy*4+xx]=site<th*tw && sy>=0 && sy<h && sx>=0 && sx<width ? __half2float(x[(sy*width+sx)*CI+c]):0.f;
            }
        }
        #pragma unroll
        for(int c2=0;c2<4;c2++) {
            b[c2]=d[c2]-d[8+c2];b[4+c2]=d[4+c2]+d[8+c2];
            b[8+c2]=d[8+c2]-d[4+c2];b[12+c2]=d[4+c2]-d[12+c2];
        }
        #pragma unroll
        for(int row=0;row<4;row++) {
            v[row*4]=b[row*4]-b[row*4+2];v[row*4+1]=b[row*4+1]+b[row*4+2];
            v[row*4+2]=b[row*4+2]-b[row*4+1];v[row*4+3]=b[row*4+1]-b[row*4+3];
        }
        #pragma unroll
        for(int j=0;j<16;j++)transformed[(j*TILES+t)*CI+c]=__float2half(v[j]);
    }
    __syncthreads();
    const int site=blockIdx.x*TILES+tile;
    float2 m[16];
    #pragma unroll
    for(int j=0;j<16;j++) {
        float2 sum=make_float2(0.f,0.f);
        #pragma unroll
        for(int chunk=0;chunk<CI;chunk+=16) {
            half2 acc=__float2half2_rn(0.f);
            #pragma unroll
            for(int k=0;k<16;k++) {
                half v=transformed[(j*TILES+tile)*CI+chunk+k];
                acc=__hfma2(__halves2half2(v,v),weights[(j*CI+chunk+k)*OP+lane],acc);
            }
            float2 v=__half22float2(acc);sum.x+=v.x;sum.y+=v.y;
        }
        m[j]=sum;
    }
    float2 a[8];
    #pragma unroll
    for(int j=0;j<4;j++) {
        a[j]=make_float2(m[j].x+m[j+4].x+m[j+8].x,m[j].y+m[j+4].y+m[j+8].y);
        a[j+4]=make_float2(m[j+4].x-m[j+8].x-m[j+12].x,m[j+4].y-m[j+8].y-m[j+12].y);
    }
    #pragma unroll
    for(int p=0;p<4;p++) {
        const int offset=(p/2)*4;
        float2 v=p%2==0?make_float2(a[offset].x+a[offset+1].x+a[offset+2].x,a[offset].y+a[offset+1].y+a[offset+2].y)
            :make_float2(a[offset+1].x-a[offset+2].x-a[offset+3].x,a[offset+1].y-a[offset+2].y-a[offset+3].y);
        v.x+=bias[2*lane];v.y+=bias[2*lane+1];
        v.x=v.x>=0.f?v.x:v.x*.1f;v.y=v.y>=0.f?v.y:v.y*.1f;
        reinterpret_cast<half2*>(refined)[(tile*4+p)*32+lane]=__floats2half2_rn(v.x,v.y);
    }
    __syncthreads();
    if constexpr(!FINAL) { if(site>=th*tw || lane>=CO/2)return; }
    if(lane<CO/2) {
    #pragma unroll
    for(int p=0;p<4;p++) {
        float2 sum=make_float2(tail_bias[2*lane],tail_bias[2*lane+1]);
        #pragma unroll
        for(int chunk=0;chunk<64;chunk+=16) {
            half2 part=__float2half2_rn(0.f);
            #pragma unroll
            for(int k=0;k<16;k++) {
                const half v=refined[(tile*4+p)*64+chunk+k];
                part=__hfma2(__halves2half2(v,v),tail_weights[(chunk+k)*(CO/2)+lane],part);
            }
            const float2 v=__half22float2(part);sum.x+=v.x;sum.y+=v.y;
        }
        const int y=site/tw*2+p/2,xx=site%tw*2+p%2;
        if(lane<CO/2) {
            if constexpr(FINAL) reinterpret_cast<half2*>(metadata)[(tile*4+p)*(CO/2)+lane]=__floats2half2_rn(sum.x,sum.y);
            else if(y<h && xx<width)out[(y*width+xx)*(CO/2)+lane]=__floats2half2_rn(sum.x,sum.y);
        }
    }
    }
    if constexpr(FINAL) {
        __syncthreads();
        if(tid<TILES*16) {
            const int cell=tid/4,parity=tid%4,t=cell/4,p=cell%4,s=blockIdx.x*TILES+t;
            const int y=(s/tw*2+p/2)*2+parity/2-py,xx=(s%tw*2+p%2)*2+parity%2-px;
            if(s<th*tw && y>=0 && y<sh && xx>=0 && xx<sw)
                reconstruct_local<MODE>(src,metadata+cell*CO,base,bank,rgb,sh,sw,y,xx,parity);
        }
    }
}

torch::Tensor finish_rgb(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor tail_w,torch::Tensor tail_bias,
                          torch::Tensor src,torch::Tensor base,torch::Tensor bank,int py,int px,int mode) {
    const int h=x.size(2),width=x.size(3),tiles=((h+1)/2)*((width+1)/2);
    auto out=torch::empty({1,3,src.size(2)*2,src.size(3)*2},src.options());
    const auto stream=at::cuda::getCurrentCUDAStream();
    #define FINISH_RGB(C,M) refine_tail<C,true,M><<<(tiles+3)/4,dim3(32,4),0,stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),reinterpret_cast<const half2*>(w.data_ptr<at::Half>()),bias.data_ptr<float>(),reinterpret_cast<const half2*>(tail_w.data_ptr<at::Half>()),tail_bias.data_ptr<float>(),nullptr,h,width,src.data_ptr<float>(),base.data_ptr<float>(),bank.data_ptr<float>(),out.data_ptr<float>(),src.size(2),src.size(3),py,px)
    if(mode==0){FINISH_RGB(64,0);}else if(mode==1){FINISH_RGB(48,1);}else if(mode==2){FINISH_RGB(48,2);}
    else TORCH_CHECK(false,"Unsupported reconstruction mode");
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor finish_controller(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor tail_w,torch::Tensor tail_bias,int co) {
    const int h=x.size(2),width=x.size(3),tiles=((h+1)/2)*((width+1)/2);
    auto out=torch::empty({1,co,h,width},x.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto stream=at::cuda::getCurrentCUDAStream();
    #define FINISH(C) refine_tail<C><<<(tiles+3)/4,dim3(32,4),0,stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),reinterpret_cast<const half2*>(w.data_ptr<at::Half>()),bias.data_ptr<float>(),reinterpret_cast<const half2*>(tail_w.data_ptr<at::Half>()),tail_bias.data_ptr<float>(),reinterpret_cast<half2*>(out.data_ptr<at::Half>()),h,width)
    if(co==64){FINISH(64);}else if(co==48){FINISH(48);}else TORCH_CHECK(false,"Unsupported tail width");
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

#include "int8_kernels.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("conv",&conv);m.def("head",&head);m.def("winograd",&winograd);m.def("analyze",&analyze);
    m.def("shuffle_conv",&shuffle_conv);m.def("project_source",&project_source);m.def("finish_controller",&finish_controller);
    m.def("quantized_body",&quantized_body);
    m.def("quantized_pair",&quantized_pair);
    m.def("project_source_int8",&project_source_int8);
    m.def("full_body",&full_body);m.def("finish_int8",&finish_int8);
    m.def("expand_to_int8",&expand_to_int8);
    m.def("finish_rgb",&finish_rgb);
    m.def("shared_head",&shared_head);
}
