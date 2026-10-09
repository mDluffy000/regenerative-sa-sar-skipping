"""CPU/CUDA migration equivalence, without reading test data."""
from pathlib import Path
import copy,json,torch
from torch import nn
from model import configure_runtime,pretrained,QuantOp
from hardware import Hardware
from data import read_data,preprocess,create_split
R=Path(__file__).resolve().parent
configure_runtime();torch.manual_seed(20260910);torch.use_deterministic_algorithms(True)
checks=[]
for kind in ['linear','conv']:
 base=nn.Linear(129,65) if kind=='linear' else nn.Conv2d(3,65,3,padding=1)
 cpu=QuantOp('fixture',base,1.);gpu=copy.deepcopy(cpu).cuda()
 x=torch.randint(-7,8,(2,3,129) if kind=='linear' else (2,3,4,4)).float()
 for mode in ['digital','exact','adc','disturbance','sa']:
  hs=[];outs=[]
  for op,inp in [(cpu,x),(gpu,x.cuda())]:
   hw=Hardware(mode);hw.context([100,101],20260910);hw.capture=True;op.hw=hw
   qx,qw,_,_=op.integers(inp)
   acc=hw.accumulator(op,qx.detach(),qw.detach()) if mode!='digital' else None
   if mode=='exact':
    direct=torch.nn.functional.linear(qx,qw) if kind=='linear' else torch.nn.functional.conv2d(qx,qw,padding=1)
    assert torch.equal(acc,direct)
   z=op(inp);z.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in op.parameters())
   hs.append(hw);outs.append(z.detach().cpu())
  torch.testing.assert_close(outs[0],outs[1],atol=2e-5,rtol=1e-5)
  assert hs[0].metrics()==hs[1].metrics()
  import numpy as np
  for a,b in zip(hs[0].traces,hs[1].traces):
   assert np.array_equal(a['raw'],b['raw']) and np.array_equal(a['gate'],b['gate'])
   assert np.array_equal(a['trace'],b['trace'],equal_nan=True)
  checks.append(dict(kind=kind,mode=mode,max_output_error=float((outs[0]-outs[1]).abs().max()),ordered_trace_and_counters_identical=True))
x,_=read_data('train');inp=preprocess(x[create_split()['train'][:2]])
m=pretrained().eval()
with torch.no_grad():
 cpu=m(inp);gpu=m.cuda()(inp.cuda()).cpu()
torch.testing.assert_close(cpu,gpu,atol=1e-4,rtol=1e-4)
assert torch.equal(cpu.argmax(1),gpu.argmax(1))
out=dict(status='PASS',checks=checks,pretrained_cpu_cuda_max_logit_error=float((cpu-gpu).abs().max()),test_data_used=False)
(R/'cloud_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
