"""Signed W4A4, tiled 128x128, one logical ADC service domain per array read.

Dynamic attention matmuls and normalization are digital. No test-derived B noise.
"""
from pathlib import Path
import ctypes as ct,subprocess
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parent
PILOT_TARGETS=['blocks.0.mlp.fc2','blocks.5.mlp.fc2','blocks.11.mlp.fc2']
COUNTERS=['requests','gated','conversions','sa_calls','near','comparisons','code_errors','analog_code_errors','prefix_code_errors','clipped_raw','absolute_code_error']
class Hardware:
 def __init__(self,mode='adc',skip=5,theta=8.,sigma=.03):
  cpp=R/'events.cpp';so=R/'events.so'
  if not so.exists() or so.stat().st_mtime<cpp.stat().st_mtime:subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
  self.lib=ct.CDLL(str(so));self.lib.convert.argtypes=[ct.c_void_p]*3+[ct.c_int]*7+[ct.c_uint64]+[ct.c_int]*2+[ct.c_double]*2+[ct.c_void_p]*3
  self.lib.convert.restype=None;self.mode=mode;self.skip=skip;self.theta=theta;self.sigma=sigma;self.ids=None;self.seed=0;self.names={};self.capture=False;self.traces=[];self.reset_stats()
 def reset_stats(self):self.stats=np.zeros(11,np.int64);self.by_layer={}
 def context(self,ids,seed):self.ids=np.asarray(ids,dtype=np.int64);self.seed=seed
 def convert(self,raw,gate,spatial,layer,rt,ctile,bit):
  raw=np.ascontiguousarray(raw,np.float32);gate=np.ascontiguousarray(gate,np.uint8);out=np.zeros_like(raw);trace=np.full((*raw.shape,9),np.nan) if self.capture else None
  ptr=lambda a:None if a is None else a.ctypes.data
  self.lib.convert(ptr(raw),ptr(gate),ptr(self.ids),len(raw),spatial,raw.shape[1],layer,rt,ctile,bit,self.seed,{'exact':0,'adc':1,'disturbance':2,'sa':3}[self.mode],self.skip,self.theta,self.sigma,ptr(out),ptr(self.stats),ptr(trace))
  if self.capture:self.traces.append(dict(layer=layer,rowtile=rt,coltile=ctile,activation_bit=bit,ids=self.ids.copy(),tokens=spatial,raw=raw.copy(),gate=gate.copy(),trace=trace))
  return torch.from_numpy(out)
 @torch.no_grad()
 def accumulator(self,op,qx,qw):
  before=self.stats.copy();device=qx.device;qx=qx.cpu();qw=qw.cpu();b=len(qx)
  if op.isconv:
   patches=F.unfold(qx,op.weight.shape[-2:],padding=op.padding,stride=op.stride,dilation=op.dilation).transpose(1,2)
  else:patches=qx.reshape(b,-1,qx.shape[-1])
  tokens=patches.shape[1];a=patches.reshape(b*tokens,-1).to(torch.int32);weights=qw.flatten(1).to(torch.int32);c=len(weights)
  assert self.ids is not None and len(self.ids)==b
  assert int(a.min())>=-8 and int(a.max())<=7
  out=torch.zeros((len(a),c),dtype=torch.float32);layer=self.names.get(op.name,0)
  for rt,start in enumerate(range(0,a.shape[1],128)):
   x=a[:,start:start+128];v=weights[:,start:start+128]+7
   # Offset weight encoding: W = (d0+4*d1)-7. Digital correction is exact.
   out-=7*x.sum(1).float()[:,None]
   for ctile,first in enumerate(range(0,c,64)):
    tile=v[first:first+64];digits=torch.stack([tile&3,(tile>>2)&3],1).reshape(2*len(tile),-1).float();dead=digits.sum(1)==0
    for bit,significance in enumerate([1,2,4,-8]):
     xb=((x>>bit)&1).float();raw=xb@digits.T;gate=(xb.sum(1)==0)[:,None]|dead[None,:]
     codes=self.convert(raw.numpy(),gate.numpy(),tokens,layer,rt,ctile,bit).reshape(len(a),len(tile),2)
     scale=1 if self.mode=='exact' else 192/255
     out[:,first:first+len(tile)]+=(codes[:,:,0]+4*codes[:,:,1])*scale*significance
  self.by_layer[op.name]=self.by_layer.get(op.name,np.zeros(11,np.int64))+self.stats-before
  if op.isconv:
   h=(qx.shape[-2]+2*op.padding[0]-op.dilation[0]*(op.weight.shape[-2]-1)-1)//op.stride[0]+1
   w=(qx.shape[-1]+2*op.padding[1]-op.dilation[1]*(op.weight.shape[-1]-1)-1)//op.stride[1]+1
   return out.reshape(b,h,w,c).permute(0,3,1,2).to(device)
  return out.reshape(*qx.shape[:-1],c).to(device)
 def metrics(self):
  d=dict(zip(COUNTERS,map(int,self.stats)));d['comparison_saving']=1-d['comparisons']/max(1,8*d['conversions']) if d['conversions'] else 0
  d['layers']={k:dict(zip(COUNTERS,map(int,v))) for k,v in self.by_layer.items()};return d
def attach(model,hw,targets=PILOT_TARGETS):
 from model import QuantOp
 allops=[(n,m) for n,m in model.named_modules() if isinstance(m,QuantOp)]
 hw.names={n:i for i,(n,m) in enumerate(allops)}
 for n,m in allops:m.hw=hw if n in targets else None
 assert set(targets)<=set(hw.names)
