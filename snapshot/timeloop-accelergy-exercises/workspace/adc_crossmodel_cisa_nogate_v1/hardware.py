import sys,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from gpu_parent import Hardware as Parent
R=Path(__file__).resolve().parent
EXT=None
META=['valid_detector_calls','rail_bypass_calls','held_d_edge_extrapolation_calls','held_CM_edge_extrapolation_calls','threshold_bracket_calls']
class Adapter:
 def __init__(self,owner):self.owner=owner
 def convert(self,*args):
  o=self.owner;out,s,tr=EXT.convert(*args,o.table,o.disturbance,o.lower,o.upper);o.meta+=s[11:];return out,s[:11],tr
class Hardware(Parent):
 def __init__(self,mode='sa',target=20,skip=6,disturbance=True):
  self.mode=mode;self.skip=skip;self.target=target;self.disturbance=disturbance;self.theta=(19 if target==20 else 38)/1000;self.sigma=0.;self.names={}
  self.lower,self.upper=(.016,.024) if target==20 else (.036,.044)
  self.table=torch.from_numpy(__import__('numpy').fromfile(R/f'held_{target}.bin',dtype='<f8').copy()).cuda();self.lib=Adapter(self);self.reset_stats()
 def reset_stats(self):
  super().reset_stats();self.meta=torch.zeros(5,dtype=torch.int64,device='cuda');self.sparsity=torch.zeros(5,dtype=torch.int64,device='cuda')
 def metrics(self):
  d=super().metrics();v=self.meta.cpu().tolist();sp=self.sparsity.cpu().tolist()
  assert d['gated']==0,'Only padding may be excluded; no data-dependent gating'
  if self.mode!='exact':assert d['conversions']==d['requests']
  d['CISA']=dict(zip(META,v));d['CISA'].update(target_mV=self.target,skip=self.skip,held_injected=self.disturbance,modeled_detector_subtotal_upper_pJ=v[0]*(.057100 if self.target==20 else .061309))
  d['sparsity_diagnostic']=dict(zip(['potential_zero_gated_events_NOT_skipped','activation_zeros','activation_elements','weight_zeros_repeated_per_call','weight_elements_repeated_per_call'],sp))
  d['SA_hit_fraction_of_all_conversions']=d['near']/max(1,d['conversions']);d['whole_ADC_conversion_skip_fraction']=0.
  return d
 @torch.no_grad()
 def accumulator(self,op,qx,qw):
  if op.isconv:
   patches=F.unfold(qx,op.weight.shape[-2:],padding=op.padding,stride=op.stride,dilation=op.dilation).transpose(1,2)
  else:patches=qx.reshape(len(qx),-1,qx.shape[-1])
  b,tokens,k=patches.shape;c=len(qw);nr=math.ceil(k/128);nc=math.ceil(c/64);reads=b*tokens
  a=patches.reshape(reads,k).to(torch.int32);w=qw.flatten(1).to(torch.int32)
  assert self.ids.numel()==b
  assert not op.isconv or op.groups==1
  # Pad activation with zero and encoded weights with zero digits. Extra columns
  # are gated then excluded from architectural request counts below.
  x=F.pad(a,(0,nr*128-k)).reshape(reads,nr,128).permute(1,0,2).contiguous()
  v=F.pad(w+7,(0,nr*128-k,0,nc*64-c)).reshape(nc*64,nr,128).permute(1,0,2)
  digits=torch.stack([v&3,(v>>2)&3],2).reshape(nr,nc*128,128).float()
  xb=((x[:,None]>>torch.arange(4,device=x.device)[None,:,None,None])&1).float()
  raw=torch.matmul(xb,digits[:,None].transpose(-1,-2)).reshape(nr,4,reads,nc,128).permute(0,1,3,2,4).contiguous()
  known_empty=(xb.sum(-1)==0);known_dead=(digits.sum(-1)==0).reshape(nr,nc,128)
  physical=(torch.arange(nc*128,device=x.device)<c*2).reshape(1,nc,128)
  ecounts=known_empty.sum((1,2));dcounts=(known_dead&physical).sum((1,2))
  potential=(ecounts*(c*2)+(4*reads-ecounts)*dcounts).sum()
  self.sparsity+=torch.stack([potential,(qx==0).sum(),torch.as_tensor(qx.numel(),device=x.device),(qw==0).sum(),torch.as_tensor(qw.numel(),device=x.device)]).to(torch.int64)
  empty=torch.zeros_like(known_empty,dtype=torch.uint8).contiguous()
  dead=(~physical).expand(nr,-1,-1).to(torch.uint8).contiguous()
  codes,s,_=self.lib.convert(raw,empty,dead,self.ids,tokens,self.names.get(op.name,0),self.seed,{'exact':0,'adc':1,'disturbance':2,'sa':3}[self.mode],self.skip,self.theta,self.sigma,False)
  padding_events=nr*4*reads*(nc*128-c*2);s[0]-=padding_events;s[1]-=padding_events
  self.stats+=s;self.by_layer[op.name]=self.by_layer.get(op.name,torch.zeros_like(s))+s
  codes=codes.permute(0,1,3,2,4).reshape(nr,4,reads,nc*64,2)[:,:,:,:c]
  if hasattr(op,"low"):codes=codes.double()
  out=torch.zeros((reads,c),device=qx.device,dtype=torch.float64 if hasattr(op,"low") else torch.float32);scale=1 if self.mode=='exact' else 192/255
  for rt in range(nr):
   out-=7*x[rt].sum(1).float()[:,None]
   for bit,significance in enumerate([1,2,4,8] if getattr(op,"low",-7)==0 else [1,2,4,-8]):out+=(codes[rt,bit,:,:,0]+4*codes[rt,bit,:,:,1])*scale*significance
  if op.isconv:
   h=(qx.shape[-2]+2*op.padding[0]-op.dilation[0]*(op.weight.shape[-2]-1)-1)//op.stride[0]+1
   ww=(qx.shape[-1]+2*op.padding[1]-op.dilation[1]*(op.weight.shape[-1]-1)-1)//op.stride[1]+1
   # Match the digital Conv2d NCHW layout as well as values. Otherwise later
   # CUDA BatchNorm can choose a different floating kernel at rounding edges.
   return out.reshape(b,h,ww,c).permute(0,3,1,2).contiguous()
  return out.reshape(*qx.shape[:-1],c)
