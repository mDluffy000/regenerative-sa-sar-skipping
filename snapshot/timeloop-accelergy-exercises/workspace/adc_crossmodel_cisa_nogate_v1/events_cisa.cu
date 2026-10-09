#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
__device__ int adc(double x){int q=(int)nearbyint(x);return q<0?0:(q>255?255:q);}
__device__ double interpolate(const double* b,double cm,double d){
 cm=fmin(fmax(cm,b[0]),b[4]);d=fmin(fmax(d,b[5]),b[9]);
 int i=0,j=0;while(i<3&&cm>b[i+1])i++;while(j<3&&d>b[6+j])j++;
 double a=(cm-b[i])/(b[i+1]-b[i]),v=(d-b[5+j])/(b[6+j]-b[5+j]);
 const double* z=b+10;return (1-a)*((1-v)*z[i*5+j]+v*z[i*5+j+1])+a*((1-v)*z[(i+1)*5+j]+v*z[(i+1)*5+j+1]);
}
// Parallel only across independent reads; ordered reference and code feedback inside each read.
__global__ void events(const float* raw,const uint8_t* empty,const uint8_t* dead,float* out,int64_t* stats,double* trace,
 int nr,int nc,int reads,int mode,int skip,double threshold,const double* b,bool disturbance,double lower,double upper){
 int64_t domain=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(domain>=(int64_t)nr*4*nc*reads)return;
 int r=domain%reads,t=domain/reads,ct=t%nc;t/=nc;int bit=t%4,rt=t/4;
 int previous=0,active=0;double reference=0;int64_t s[16]={0};
 for(int c=0;c<128;c++){
  int64_t ix=domain*128+c;s[0]++;out[ix]=0;
  if(empty[((int64_t)rt*4+bit)*reads+r]||dead[((int64_t)rt*nc+ct)*128+c]){s[1]++;continue;}
  if(mode==0){out[ix]=raw[ix];active++;continue;}
  double v=raw[ix]/160.0,cm=(v+reference)/2,d=v-reference,e=0;bool call=active>0&&mode>=2;
  bool valid=call&&v>=0&&v<=1.2&&reference>=0&&reference<=1.2,near=false;
  if(call){s[11]+=valid;s[12]+=!valid;}
  if(valid){
   s[13]+=(d<b[5]||d>b[9]);s[14]+=(cm<b[0]||cm>b[4]);s[15]+=(fabs(d)>lower&&fabs(d)<upper);
   if(disturbance)e=interpolate(b,cm,d);
   near=mode==3&&fabs(d)<=threshold;
  }
  double held=v+e;int gold=adc(raw[ix]*255.0/192.0),full=e==0?gold:adc(raw[ix]*255.0/192.0+e*212.5);
  int lo=near?((previous>>(8-skip))<<(8-skip)):0,hi=near?lo+(1<<(8-skip))-1:255;
  int q=full<lo?lo:(full>hi?hi:full);out[ix]=q;
  s[2]++;s[3]+=call;s[4]+=near;s[5]+=8-(near?skip:0);s[6]+=q!=gold;s[7]+=full!=gold;s[8]+=q!=full;s[9]+=raw[ix]>192;s[10]+=abs(q-gold);
  if(trace){double* z=trace+ix*12;z[0]=previous;z[1]=v;z[2]=reference;z[3]=cm;z[4]=d;z[5]=e;z[6]=held;z[7]=near;z[8]=gold;z[9]=full;z[10]=q;z[11]=8-(near?skip:0);}
  previous=q;reference=held;active++;
 }
 for(int i=0;i<16;i++)atomicAdd((unsigned long long*)(stats+i),(unsigned long long)s[i]);
}
std::vector<torch::Tensor> convert(torch::Tensor raw,torch::Tensor empty,torch::Tensor dead,torch::Tensor ids,
 int spatial,int layer,int64_t seed,int mode,int skip,double theta,double sigma,bool capture,torch::Tensor table,bool disturbance,double lower,double upper){
 TORCH_CHECK(sigma==0&&skip>=4&&skip<=6,"skip4..6 deterministic model");
 TORCH_CHECK(table.is_cuda()&&table.scalar_type()==torch::kFloat64&&table.is_contiguous()&&table.numel()==35,"table");
 TORCH_CHECK(raw.is_cuda()&&raw.is_contiguous()&&raw.scalar_type()==torch::kFloat32&&raw.dim()==5&&raw.size(1)==4&&raw.size(4)==128,"raw");
 TORCH_CHECK(empty.is_cuda()&&empty.is_contiguous()&&empty.scalar_type()==torch::kUInt8&&dead.is_cuda()&&dead.is_contiguous()&&dead.scalar_type()==torch::kUInt8,"gates");
 int nr=raw.size(0),nc=raw.size(2),reads=raw.size(3);TORCH_CHECK(spatial>0&&ids.numel()*spatial==reads&&empty.numel()==nr*4*reads&&dead.numel()==nr*nc*128,"shapes");
 c10::cuda::CUDAGuard guard(raw.device());auto out=torch::empty_like(raw);auto stats=torch::zeros({16},raw.options().dtype(torch::kInt64));
 auto tr=capture?torch::full({nr,4,nc,reads,128,12},NAN,raw.options().dtype(torch::kFloat64)):torch::empty({0},raw.options().dtype(torch::kFloat64));
 int64_t domains=(int64_t)nr*4*nc*reads;
 events<<<(domains+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(raw.data_ptr<float>(),empty.data_ptr<uint8_t>(),dead.data_ptr<uint8_t>(),out.data_ptr<float>(),stats.data_ptr<int64_t>(),capture?tr.data_ptr<double>():nullptr,nr,nc,reads,mode,skip,theta,table.data_ptr<double>(),disturbance,lower,upper);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,stats,tr};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("convert",&convert);}
