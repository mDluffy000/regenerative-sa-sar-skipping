import sys,json,time,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from wrn import WRN,QuantOp,runtime,prep,quantize,attach,BASE,TARGETS
from gpu_hardware import Hardware,extension
from hardware import Hardware as CPU
from data import read_data,create_split
R=Path(__file__).resolve().parent
runtime();torch.manual_seed(20260910)

def kernel_check():
 # Boundaries, zero gating, multiple reduction/output tiles and sample IDs.
 g=torch.Generator().manual_seed(923);raw=torch.randint(0,385,(3,4,4,128,128),generator=g).float()
 empty=(torch.rand((3,4,128),generator=g)<.1).byte();dead=(torch.rand((3,4,128),generator=g)<.1).byte()
 ids=np.array([100,205],np.int64);results=[]
 for mode,sigma in [('adc',.03),('disturbance',.03),('sa',.03),('sa',0.)]:
  code,s,tr=extension().convert(raw.cuda(),empty.cuda(),dead.cuda(),torch.from_numpy(ids).cuda(),64,17,20260910,{'adc':1,'disturbance':2,'sa':3}[mode],5,8.,sigma,True)
  cpu=CPU(mode,sigma=sigma);cpu.context(ids,20260910);cpu.capture=True
  ref=torch.zeros_like(raw);trs=[]
  for rt in range(3):
   for bit in range(4):
    for ct in range(4):
     gate=empty[rt,bit,:,None].bool()|dead[rt,ct,None,:].bool()
     ref[rt,bit,ct]=cpu.convert(raw[rt,bit,ct].numpy(),gate.numpy(),64,17,rt,ct,bit)
  assert torch.equal(code.cpu(),ref)
  assert np.array_equal(s.cpu().numpy(),cpu.stats)
  reftr=np.stack([v['trace'] for v in cpu.traces]).reshape(*tr.shape)
  np.testing.assert_allclose(tr.cpu().numpy(),reftr,rtol=0,atol=1e-11,equal_nan=True)
  results.append(dict(mode=mode,sigma=sigma,events=raw.numel(),code_mismatches=0,counter_mismatches=0))
 return results

def accumulation_check():
 out=[]
 for conv,k,c,t in [(False,3,3,2),(False,129,65,2),(False,257,129,1),(True,3,65,4)]:
  op=QuantOp('fixture',nn.Conv2d(k,c,3,padding=1) if conv else nn.Linear(k,c),1.)
  x=torch.randint(-8,8,(2,k,t,t) if conv else (2,t,k)).float();w=torch.randint(-7,8,op.weight.shape).float();x[0]=0;w[0]=-7
  for mode in ['exact','adc','disturbance','sa']:
   a=CPU(mode);a.context([9,25],20260910);b=Hardware(mode);b.context([9,25],20260910)
   z=a.accumulator(op,x,w);zz=b.accumulator(op,x.cuda(),w.cuda()).cpu()
   assert torch.equal(z,zz),(conv,k,c,mode,float((z-zz).abs().max()))
   assert a.metrics()==b.metrics()
   if mode=='exact':assert torch.equal(z,F.conv2d(x,w,padding=1) if conv else F.linear(x,w))
   out.append(dict(conv=conv,k=k,c=c,mode=mode,max_error=0))
 return out

def functional_reference(m,x):
 def conv(x,op):return F.conv2d(x,op.weight,None,op.stride,op.padding)
 def bn(x,op):return F.batch_norm(x,op.running_mean,op.running_var,op.weight,op.bias,False,0.,op.eps)
 x=conv(x,m.conv)
 for group in m.groups:
  for b in group:
   y=F.relu(bn(x,b.bn1));z=conv(F.relu(bn(conv(y,b.conv1),b.bn2)),b.conv2)
   x=z+(conv(y,b.shortcut) if b.shortcut is not None else x)
 x=F.avg_pool2d(F.relu(bn(x,m.bn)),8).flatten(1)
 return F.linear(x,m.fc.weight,m.fc.bias)

def main():
 kernels=kernel_check();acc=accumulation_check();print('ORDERED_KERNEL_AND_MAPPING_PASS',flush=True)
 m=WRN().cuda().eval();x,y=read_data('train');ids=np.array(create_split()['train'][:128]);inp=prep(x[ids[:2]])
 with torch.no_grad():a=m(inp);b=functional_reference(m,inp)
 torch.testing.assert_close(a,b,rtol=0,atol=0)
 params=sum(v.numel() for v in m.parameters());assert params==36536884,params
 samples={};hooks=[]
 for name,op in m.named_modules():
  if isinstance(op,(nn.Conv2d,nn.Linear)):
   def hook(op,args,n=name):
    a=args[0].detach().abs().flatten();samples.setdefault(n,[]).append(a[::max(1,a.numel()//4096)][:4096].cpu())
   hooks.append(op.register_forward_pre_hook(hook))
 with torch.no_grad():m(prep(x[ids[:8]]))
 for h in hooks:h.remove()
 scales={n:max(float(torch.quantile(torch.cat(v),.999))/7,1e-8) for n,v in samples.items()}
 q=quantize(copy.deepcopy(m),scales);hw=Hardware('exact');attach(q,hw);hw.context(ids[:1],20260910)
 with torch.no_grad():
  hw.mode='digital';a=q(inp[:1]);hw.mode='exact';b=q(inp[:1]);torch.testing.assert_close(a,b,rtol=0,atol=0)
 bench=[]
 for name,model,batch,amp,mode in [('floating_BF16_train',m,128,True,'digital'),('W4A4',q,32,False,'digital'),('W4A4_SA',q,4,False,'sa')]:
  model.train();hw.mode=mode;hw.context(ids[:batch],20260910);hw.reset_stats();a=prep(x[ids[:batch]]);yy=y[ids[:batch]].cuda();times=[]
  for repeat in range(3):
   model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z=model(a);loss=F.cross_entropy(z,yy)
   loss.backward();torch.cuda.synchronize();times.append(time.monotonic()-start);assert torch.isfinite(loss)
  bench.append(dict(route=name,batch=batch,seconds_per_image=float(np.median(times[1:]))/batch,times=times,hardware=hw.metrics()))
  print('BENCH',name,bench[-1]['seconds_per_image'],flush=True)
 out=dict(status='PASS',parameters=params,quantized_ops=29,targets=TARGETS,kernel_checks=kernels,mapping_checks=acc,whole_model_exact_error=0,topology_reference_error=0,benchmark=bench,test_data_used=False,
  cuda=torch.version.cuda,torch=torch.__version__)
 (R/'preflight.json').write_text(json.dumps(out,indent=2)+'\n');print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
