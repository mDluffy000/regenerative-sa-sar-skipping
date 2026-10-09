from pathlib import Path
import os,math
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
R=Path(__file__).resolve().parent
os.environ.setdefault('MAX_JOBS','2');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0')
ext=None
def extension():
 global ext
 if ext is None:
  (R/'build').mkdir(exist_ok=True)
  ext=load('wrn_ordered_adc_v1',sources=[str(R/'events_cuda.cu')],build_directory=str(R/'build'),extra_cuda_cflags=['-O3','--fmad=false'],extra_cflags=['-O3'],verbose=False)
 return ext
COUNTERS=['requests','gated','conversions','sa_calls','near','comparisons','code_errors','analog_code_errors','prefix_code_errors','clipped_raw','absolute_code_error']
class Hardware:
 def __init__(self,mode='adc',skip=5,theta=8.,sigma=.03):
  self.lib=extension();self.mode=mode;self.skip=skip;self.theta=theta;self.sigma=sigma;self.names={};self.reset_stats()
 def context(self,ids,seed):self.ids=torch.as_tensor(ids,dtype=torch.int64,device='cuda');self.seed=seed
 def reset_stats(self):self.stats=torch.zeros(11,dtype=torch.int64,device='cuda');self.by_layer={}
 def metrics(self):
  d=dict(zip(COUNTERS,self.stats.cpu().tolist()));d['comparison_saving']=1-d['comparisons']/max(1,8*d['conversions']) if d['conversions'] else 0
  d['layers']={n:dict(zip(COUNTERS,s.cpu().tolist())) for n,s in self.by_layer.items()};return d
 @torch.no_grad()
 def accumulator(self,op,qx,qw):
  if op.isconv:
   patches=F.unfold(qx,op.weight.shape[-2:],padding=op.padding,stride=op.stride,dilation=op.dilation).transpose(1,2)
  else:patches=qx.reshape(len(qx),-1,qx.shape[-1])
  b,tokens,k=patches.shape;c=len(qw);nr=math.ceil(k/128);nc=math.ceil(c/64);reads=b*tokens
  a=patches.reshape(reads,k).to(torch.int32);w=qw.flatten(1).to(torch.int32)
  assert self.ids.numel()==b
  # Pad activation with zero and encoded weights with zero digits. Extra columns
  # are gated then excluded from architectural request counts below.
  x=F.pad(a,(0,nr*128-k)).reshape(reads,nr,128).permute(1,0,2).contiguous()
  v=F.pad(w+7,(0,nr*128-k,0,nc*64-c)).reshape(nc*64,nr,128).permute(1,0,2)
  digits=torch.stack([v&3,(v>>2)&3],2).reshape(nr,nc*128,128).float()
  xb=((x[:,None]>>torch.arange(4,device=x.device)[None,:,None,None])&1).float()
  raw=torch.matmul(xb,digits[:,None].transpose(-1,-2)).reshape(nr,4,reads,nc,128).permute(0,1,3,2,4).contiguous()
  empty=(xb.sum(-1)==0).to(torch.uint8).contiguous();dead=(digits.sum(-1)==0).reshape(nr,nc,128).to(torch.uint8).contiguous()
  codes,s,_=self.lib.convert(raw,empty,dead,self.ids,tokens,self.names.get(op.name,0),self.seed,{'exact':0,'adc':1,'disturbance':2,'sa':3}[self.mode],self.skip,self.theta,self.sigma,False)
  padding_events=nr*4*reads*(nc*128-c*2);s[0]-=padding_events;s[1]-=padding_events
  self.stats+=s;self.by_layer[op.name]=self.by_layer.get(op.name,torch.zeros_like(s))+s
  codes=codes.permute(0,1,3,2,4).reshape(nr,4,reads,nc*64,2)[:,:,:,:c]
  out=torch.zeros((reads,c),device=qx.device);scale=1 if self.mode=='exact' else 192/255
  for rt in range(nr):
   out-=7*x[rt].sum(1).float()[:,None]
   for bit,significance in enumerate([1,2,4,-8]):out+=(codes[rt,bit,:,:,0]+4*codes[rt,bit,:,:,1])*scale*significance
  if op.isconv:
   h=(qx.shape[-2]+2*op.padding[0]-op.dilation[0]*(op.weight.shape[-2]-1)-1)//op.stride[0]+1
   ww=(qx.shape[-1]+2*op.padding[1]-op.dilation[1]*(op.weight.shape[-1]-1)-1)//op.stride[1]+1
   # Match the digital Conv2d NCHW layout as well as values. Otherwise later
   # CUDA BatchNorm can choose a different floating kernel at rounding edges.
   return out.reshape(b,h,ww,c).permute(0,3,1,2).contiguous()
  return out.reshape(*qx.shape[:-1],c)
