import copy,json
import torch
from torch import nn
import torch.nn.functional as F
from wrn import WRN,QuantOp,runtime,prep,quantize,attach,TARGETS
from gpu_hardware import Hardware
from data import read_data,create_split
runtime();torch.manual_seed(20260910)
m=WRN().cuda().eval();x,_=read_data('train');inp=prep(x[create_split()['train'][:2]])
samples={};hooks=[]
for n,op in m.named_modules():
 if isinstance(op,(nn.Conv2d,nn.Linear)):
  hooks.append(op.register_forward_pre_hook(lambda op,args,n=n:samples.update({n:max(float(args[0].detach().abs().max())/7,1e-8)})))
with torch.no_grad():m(inp)
for h in hooks:h.remove()
q=quantize(copy.deepcopy(m),samples);hw=Hardware('exact');attach(q,hw);hw.context([9],1)
orig=hw.accumulator
def wrapped(op,a,w):
 y=orig(op,a,w);direct=F.conv2d(a,w,None,op.stride,op.padding)
 print(json.dumps(dict(layer=op.name,a_noninteger=int((a!=a.round()).sum()),w_noninteger=int((w!=w.round()).sum()),shape=list(a.shape),max_difference=float((direct-y).abs().max()),digital_noninteger=int((direct!=direct.round()).sum()),cim_noninteger=int((y!=y.round()).sum()))),flush=True)
 if (direct-y).abs().max()>0:
  torch.save(dict(a=a.cpu(),w=w.cpu(),y=y.cpu(),direct=direct.cpu(),stride=op.stride,padding=op.padding),'/root/autodl-tmp/adc_wrn28_10_v1/diagnostic_fixture.pt')
 return y
hw.accumulator=wrapped
with torch.no_grad():q(inp[:1])
