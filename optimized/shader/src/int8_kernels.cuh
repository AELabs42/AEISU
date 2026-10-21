// Packed signed INT8 dot products execute DP4A on SIMT integer lanes, not MMA.
__device__ __forceinline__ int quantize_s8(float value,float inverse_scale) {
    return max(-127,min(127,__float2int_rn(value*inverse_scale)));
}

template<int CI,int CO,int KS,int GROUPS,bool COMPRESS,bool EXPAND,int PX,bool SKIP8=false,bool HALF_OUT=false,bool LINEAR=false,bool SUM8=SKIP8>
__global__ void int8_body(const int* __restrict__ x,const half* __restrict__ skip,
                          const int* __restrict__ weights,const float* __restrict__ factors,
                          const float* __restrict__ bias,void* __restrict__ output,
                          int h,int width,float inverse_in,float inverse_out,float output_scale,
                          float skip_scale=0.f,float inverse_sum=0.f) {
    constexpr int CG=CI/GROUPS,WORDS=CI/4;
    __shared__ int tile[KS*KS*PX*WORDS];
    const int channel=threadIdx.x,pixel=threadIdx.y,tid=pixel*CO+channel;
    const int site=blockIdx.x*PX+pixel,group=channel/(CO/GROUPS);
    for(int i=tid;i<KS*KS*PX*WORDS;i+=PX*CO) {
        const int word=i%WORDS,p=(i/WORDS)%PX,tap=i/(WORDS*PX);
        const int pos=blockIdx.x*PX+p;
        const int y=pos/width+tap/KS-KS/2,xx=pos%width+tap%KS-KS/2;
        int value=0;
        if(pos<h*width && y>=0 && y<h && xx>=0 && xx<width) {
            if constexpr(COMPRESS) {
                unsigned int bits=0;
                #pragma unroll
                for(int k=0;k<4;k++) {
                    const int c=word*4+k,sy=2*y+(c%4)/2,sx=2*xx+c%2;
                    const int s=(sy*width*2+sx)*32+c/4;
                    const int q=SKIP8?reinterpret_cast<const signed char*>(skip)[s]:quantize_s8(__half2float(skip[s]),inverse_in);
                    bits|=(static_cast<unsigned int>(q)&255u)<<(k*8);
                }
                value=static_cast<int>(bits);
            } else value=x[(y*width+xx)*WORDS+word];
        }
        tile[i]=value;
    }
    __syncthreads();
    if(site>=h*width)return;
    int accum=0;
    #pragma unroll
    for(int tap=0;tap<KS*KS;tap++) {
        #pragma unroll
        for(int word=0;word<CG/4;word++) {
            accum=__dp4a(tile[(tap*PX+pixel)*WORDS+group*(CG/4)+word],
                          weights[(tap*(CG/4)+word)*CO+channel],accum);
        }
    }
    float value=static_cast<float>(accum)*factors[channel]+bias[channel];
    if constexpr(LINEAR) {
        static_cast<half*>(output)[site*CO+channel]=__float2half(value);
        return;
    }
    value=value>=0.f?value:value*.1f;
    const int q=quantize_s8(value,inverse_out);
    if constexpr(EXPAND) {
        const int parity=channel%4,c=channel/4,y=site/width*2+parity/2,xx=site%width*2+parity%2;
        const int s=(y*width*2+xx)*32+c;
        const float skip_value=SKIP8?reinterpret_cast<const signed char*>(skip)[s]*skip_scale:__half2float(skip[s]);
        if constexpr(SUM8 && !HALF_OUT)
            static_cast<signed char*>(output)[s]=static_cast<signed char>(quantize_s8(q*output_scale+skip_value,inverse_sum));
        else static_cast<half*>(output)[s]=__float2half(q*output_scale+skip_value);
    } else static_cast<signed char*>(output)[site*CO+channel]=static_cast<signed char>(q);
}

torch::Tensor expand_to_int8(torch::Tensor x,torch::Tensor skip,torch::Tensor w,torch::Tensor factors,torch::Tensor bias,
                             double expand_scale,double sum_scale) {
    const int h=x.size(2),width=x.size(3);
    auto out=torch::empty({1,32,h*2,width*2},x.options().memory_format(at::MemoryFormat::ChannelsLast));
    int8_body<64,128,1,1,false,true,4,false,false,false,true><<<(h*width+3)/4,dim3(128,4),0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int*>(x.data_ptr<int8_t>()),reinterpret_cast<const half*>(skip.data_ptr<at::Half>()),
        w.data_ptr<int>(),factors.data_ptr<float>(),bias.data_ptr<float>(),out.data_ptr(),h,width,0.f,
        1.f/static_cast<float>(expand_scale),static_cast<float>(expand_scale),0.f,1.f/static_cast<float>(sum_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor full_body(torch::Tensor x,torch::Tensor skip,torch::Tensor weights,torch::Tensor factors,torch::Tensor bias,
                        int kind,double input_scale,double output_scale,double skip_scale,double sum_scale,bool half_out) {
    const int h=kind==0?skip.size(2)/2:x.size(2),width=kind==0?skip.size(3)/2:x.size(3);
    auto out=torch::empty({1,kind==2?32:64,kind==2?h*2:h,kind==2?width*2:width},
                          skip.options().dtype((kind==2 && half_out) || kind==4?torch::kFloat16:torch::kInt8)
                          .memory_format(at::MemoryFormat::ChannelsLast));
    const auto stream=at::cuda::getCurrentCUDAStream();
    const int* xp=kind==0?nullptr:reinterpret_cast<const int*>(x.data_ptr<int8_t>());
    const auto sp=reinterpret_cast<const half*>(skip.data_ptr<int8_t>());
    const float inv_in=1.f/static_cast<float>(input_scale),inv_out=1.f/static_cast<float>(output_scale);
    #define FULL_BODY(CI,CO,K,G,C,E,H) int8_body<CI,CO,K,G,C,E,4,true,H><<<(h*width+3)/4,dim3(CO,4),0,stream>>>(xp,sp,weights.data_ptr<int>(),factors.data_ptr<float>(),bias.data_ptr<float>(),out.data_ptr(),h,width,inv_in,inv_out,static_cast<float>(output_scale),static_cast<float>(skip_scale),1.f/static_cast<float>(sum_scale))
    if(kind==0){FULL_BODY(128,64,1,1,true,false,false);}
    else if(kind==1){FULL_BODY(64,64,3,2,false,false,false);}
    else if(kind==3){FULL_BODY(32,64,3,1,false,false,false);}
    else if(kind==4) {
        int8_body<64,64,1,1,false,false,4,true,false,true><<<(h*width+3)/4,dim3(64,4),0,stream>>>(
            xp,sp,weights.data_ptr<int>(),factors.data_ptr<float>(),bias.data_ptr<float>(),out.data_ptr(),h,width,
            inv_in,inv_out,static_cast<float>(output_scale));
    }
    else if(half_out){FULL_BODY(64,128,1,1,false,true,true);}
    else {FULL_BODY(64,128,1,1,false,true,false);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void refine_tail_int8(const int* __restrict__ x,const int* __restrict__ wr,const int* __restrict__ wt,
                                 const float* __restrict__ fr,const float* __restrict__ ft,
                                 const float* __restrict__ br,const float* __restrict__ bt,
                                 half* __restrict__ output,int h,int width,float inverse_refine) {
    constexpr int PX=4,CO=64,WORDS=8;
    __shared__ int tile[9*PX*WORDS];
    __shared__ int refined[PX*16];
    const int c=threadIdx.x,pixel=threadIdx.y,tid=pixel*CO+c,site=blockIdx.x*PX+pixel;
    for(int i=tid;i<9*PX*WORDS;i+=PX*CO) {
        const int word=i%WORDS,p=(i/WORDS)%PX,tap=i/(WORDS*PX),s=blockIdx.x*PX+p;
        const int y=s/width+tap/3-1,xx=s%width+tap%3-1;
        tile[i]=s<h*width && y>=0 && y<h && xx>=0 && xx<width?x[(y*width+xx)*WORDS+word]:0;
    }
    __syncthreads();
    int acc=0;
    #pragma unroll
    for(int tap=0;tap<9;tap++) {
        #pragma unroll
        for(int word=0;word<WORDS;word++)acc=__dp4a(tile[(tap*PX+pixel)*WORDS+word],wr[(tap*WORDS+word)*CO+c],acc);
    }
    float value=static_cast<float>(acc)*fr[c]+br[c];value=value>=0.f?value:value*.1f;
    reinterpret_cast<signed char*>(refined)[pixel*CO+c]=static_cast<signed char>(quantize_s8(value,inverse_refine));
    __syncthreads();
    if(site>=h*width)return;
    acc=0;
    #pragma unroll
    for(int word=0;word<16;word++)acc=__dp4a(refined[pixel*16+word],wt[word*CO+c],acc);
    output[site*CO+c]=__float2half(static_cast<float>(acc)*ft[c]+bt[c]);
}

torch::Tensor finish_int8(torch::Tensor x,torch::Tensor wr,torch::Tensor wt,torch::Tensor fr,torch::Tensor ft,
                          torch::Tensor br,torch::Tensor bt,double output_scale) {
    const int h=x.size(2),width=x.size(3);
    auto out=torch::empty({1,64,h,width},x.options().dtype(torch::kFloat16).memory_format(at::MemoryFormat::ChannelsLast));
    refine_tail_int8<<<(h*width+3)/4,dim3(64,4),0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int*>(x.data_ptr<int8_t>()),wr.data_ptr<int>(),wt.data_ptr<int>(),fr.data_ptr<float>(),ft.data_ptr<float>(),
        br.data_ptr<float>(),bt.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr<at::Half>()),h,width,1.f/static_cast<float>(output_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor quantized_body(torch::Tensor x,torch::Tensor skip,torch::Tensor weights,torch::Tensor factors,
                             torch::Tensor bias,int kind,double input_scale,double output_scale,int pixel_tile) {
    TORCH_CHECK(kind>=0 && kind<=2,"INT8 body kind must be compress, bottleneck, or expand");
    const int h=kind==0?skip.size(2)/2:x.size(2),width=kind==0?skip.size(3)/2:x.size(3);
    auto out=torch::empty({1,kind==2?32:64,kind==2?h*2:h,kind==2?width*2:width},
                          skip.options().dtype(kind==2?torch::kFloat16:torch::kInt8).memory_format(at::MemoryFormat::ChannelsLast));
    const auto stream=at::cuda::getCurrentCUDAStream();
    const int* xp=kind==0?nullptr:reinterpret_cast<const int*>(x.data_ptr<int8_t>());
    const auto sp=reinterpret_cast<const half*>(skip.data_ptr<at::Half>());
    const auto wp=weights.data_ptr<int>();
    const float inv_in=1.f/static_cast<float>(input_scale),inv_out=1.f/static_cast<float>(output_scale);
    #define INT_BODY(CI,CO,K,G,C,E,P) int8_body<CI,CO,K,G,C,E,P><<<(h*width+P-1)/P,dim3(CO,P),0,stream>>>(xp,sp,wp,factors.data_ptr<float>(),bias.data_ptr<float>(),out.data_ptr(),h,width,inv_in,inv_out,static_cast<float>(output_scale))
    if(kind==0){INT_BODY(128,64,1,1,true,false,4);}
    else if(kind==1){INT_BODY(64,64,3,2,false,false,4);}
    else {INT_BODY(64,128,1,1,false,true,4);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Fuse two spatial layers using a shared-memory halo. Quantization still occurs
// at the original boundary; out-of-image intermediate sites remain zero.
__global__ void int8_pair(const int* __restrict__ x,const int* __restrict__ w0,const int* __restrict__ w1,
                          const float* __restrict__ f0,const float* __restrict__ f1,
                          const float* __restrict__ b0,const float* __restrict__ b1,
                          signed char* __restrict__ out,int h,int width,float inv0,float inv1) {
    constexpr int SIDE=8,INPUT_SIDE=12,MID=10,WORDS=16;
    __shared__ int input[INPUT_SIDE*INPUT_SIDE*WORDS];
    __shared__ int middle[MID*MID*WORDS];
    const int c=threadIdx.x,pixel=threadIdx.y,tid=pixel*64+c;
    const int oy=blockIdx.y*SIDE,ox=blockIdx.x*SIDE;
    for(int i=tid;i<INPUT_SIDE*INPUT_SIDE*WORDS;i+=256) {
        const int site=i/WORDS,y=oy+site/INPUT_SIDE-2,xx=ox+site%INPUT_SIDE-2;
        input[i]=y>=0 && y<h && xx>=0 && xx<width?x[(y*width+xx)*WORDS+i%WORDS]:0;
    }
    __syncthreads();
    for(int site=pixel;site<MID*MID;site+=4) {
        const int ly=site/MID,lx=site%MID,group=c/32;
        int acc=0;
        #pragma unroll
        for(int tap=0;tap<9;tap++) {
            #pragma unroll
            for(int word=0;word<8;word++)
                acc=__dp4a(input[((ly+tap/3)*INPUT_SIDE+lx+tap%3)*WORDS+group*8+word],w0[(tap*8+word)*64+c],acc);
        }
        float value=static_cast<float>(acc)*f0[c]+b0[c];value=value>=0.f?value:value*.1f;
        const int y=oy+ly-1,xx=ox+lx-1;
        reinterpret_cast<signed char*>(middle)[site*64+c]=y>=0 && y<h && xx>=0 && xx<width
            ? static_cast<signed char>(quantize_s8(value,inv0)):0;
    }
    __syncthreads();
    for(int site=pixel;site<SIDE*SIDE;site+=4) {
        const int ly=site/SIDE,lx=site%SIDE,group=c/32;
        int acc=0;
        #pragma unroll
        for(int tap=0;tap<9;tap++) {
            #pragma unroll
            for(int word=0;word<8;word++)
                acc=__dp4a(middle[((ly+tap/3)*MID+lx+tap%3)*WORDS+group*8+word],w1[(tap*8+word)*64+c],acc);
        }
        float value=static_cast<float>(acc)*f1[c]+b1[c];value=value>=0.f?value:value*.1f;
        const int y=oy+ly,xx=ox+lx;
        if(y<h && xx<width)out[(y*width+xx)*64+c]=static_cast<signed char>(quantize_s8(value,inv1));
    }
}

torch::Tensor quantized_pair(torch::Tensor x,torch::Tensor w0,torch::Tensor w1,torch::Tensor f0,torch::Tensor f1,
                             torch::Tensor b0,torch::Tensor b1,double scale0,double scale1) {
    auto out=torch::empty_like(x);
    const int h=x.size(2),width=x.size(3);
    int8_pair<<<dim3((width+7)/8,(h+7)/8),dim3(64,4),0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int*>(x.data_ptr<int8_t>()),w0.data_ptr<int>(),w1.data_ptr<int>(),f0.data_ptr<float>(),
        f1.data_ptr<float>(),b0.data_ptr<float>(),b1.data_ptr<float>(),out.data_ptr<int8_t>(),h,width,
        1.f/static_cast<float>(scale0),1.f/static_cast<float>(scale1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
