from pathlib import Path
import copy,hashlib,json,time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from model import pretrained,reference_torchvision,quantize,QuantOp,configure_runtime
from hardware import Hardware,attach,PILOT_TARGETS
from data import read_data,preprocess,create_split,subset
R=Path(__file__).resolve().parent
def save(n,d):(R/n).write_text(json.dumps(d,indent=2)+'\n')
def calibrate(m,x,ids):
 samples={};hooks=[]
 def hook(name,op,args):
  a=args[0].detach().abs().flatten();step=max(1,a.numel()//4096);samples.setdefault(name,[]).append(a[::step][:4096].cpu())
 for n,op in m.named_modules():
  if isinstance(op,(nn.Conv2d,nn.Linear)):hooks.append(op.register_forward_pre_hook(lambda op,args,n=n:hook(n,op,args)))
 with torch.inference_mode():
  for i in range(0,len(ids),8):m(preprocess(x[ids[i:i+8]]).to(next(m.parameters()).device))
 for h in hooks:h.remove()
 return {n:max(float(torch.quantile(torch.cat(v),.999))/7,1e-8) for n,v in samples.items()}
def mapping_checks():
 torch.manual_seed(3100);checks=[]
 for k,c,t in [(3,3,2),(129,65,2),(257,129,1)]:
  base=nn.Linear(k,c);op=QuantOp('fixture',base,1.)
  x=torch.randint(-8,8,(2,t,k)).float();w=torch.randint(-7,8,(c,k)).float();x[0,0]=0;w[0]=-7
  hw=Hardware('exact');hw.context([100,101],20260910);exact=hw.accumulator(op,x,w);direct=F.linear(x,w)
  assert torch.equal(exact,direct),(exact-direct).abs().max()
  hw.mode='sa';hw.capture=True;hw.reset_stats();hw.accumulator(op,x,w)
  checked=0
  for tr in hw.traces:
   for r in range(len(tr['raw'])):
    oldq=0;held_old=0;active=0
    for col,raw in enumerate(tr['raw'][r]):
     if tr['gate'][r,col]:continue
     ev=tr['trace'][r,col];held=float(raw)*255/192+ev[1]
     near=active>0 and abs(held-held_old)<=hw.theta
     assert bool(ev[3])==near and ev[0]==oldq
     n=8-(hw.skip if near else 0);q=(oldq>>n)<<n
     for bit in range(n-1,-1,-1):
      trial=q|(1<<bit)
      if held>trial-.5 or (held==trial-.5 and trial%2==0):q=trial
     assert q==ev[6],(q,ev);oldq=q;held_old=held;active+=1;checked+=1
  met=hw.metrics();assert met['requests']==met['gated']+met['conversions'];assert met['comparisons']==8*met['conversions']-hw.skip*met['near']
  checks.append(dict(reduction=k,outputs=c,tokens=t,exact_integer_elements=direct.numel(),ordered_scalar_sar_events=checked,integer_mismatch=0))
 # Public deterministic fixture for ordering and returned-code feedback.
 tr=hw.traces[0];np.savez_compressed(R/'mapping_trace_fixture.npz',**tr)
 return checks
def main():
 configure_runtime();torch.manual_seed(20260910);torch.use_deterministic_algorithms(True)
 split=create_split();x,y=read_data('train');m=pretrained().cuda().eval();ref=reference_torchvision(m.state_dict()).cuda().eval();inp=preprocess(x[split['train'][:2]]).cuda()
 with torch.inference_mode():a=m(inp);b=ref(inp)
 torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-5);assert torch.equal(a.argmax(1),b.argmax(1))
 checks=mapping_checks()
 m.head=nn.Linear(192,100).cuda();nn.init.trunc_normal_(m.head.weight,std=.02);nn.init.zeros_(m.head.bias)
 calids=subset(y,split['train'],1,20260910) # 100 training images, fixed before any task score.
 scales=calibrate(m,x,calids);save('initial_calibration.json',{'ids':calids.tolist(),'source':'official train only','method':'99.9 percentile abs, deterministic strided sample; signed A [-7,7]','scales':scales})
 q=quantize(copy.deepcopy(m),scales)
 hw=Hardware('sa');attach(q,hw)
 bench=[]
 for name,model,batch in [('FP32',m,8),('W4A4_digital',q,4),('W4A4_native_ADC',q,2),('W4A4_SA_skip5',q,2)]:
  hw.mode={'FP32':'digital','W4A4_digital':'digital','W4A4_native_ADC':'adc','W4A4_SA_skip5':'sa'}[name]
  ids=calids[:batch];hw.context(ids,20260910);hw.reset_stats();model.train();zinput=preprocess(x[ids]).cuda();ts=[];lossvalue=None
  for repeat in range(3):
   model.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.monotonic();z=model(zinput);loss=F.cross_entropy(z,y[ids].cuda());loss.backward();torch.cuda.synchronize();ts.append(time.monotonic()-begin);lossvalue=float(loss.detach())
   assert torch.isfinite(loss)
  if model is q:
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
   assert all(op.activation_scale.grad is not None for op in model.modules() if isinstance(op,QuantOp))
  bench.append({'route':name,'batch':batch,'forward_backward_seconds':ts,'steady_seconds_per_image':float(np.median(ts[1:]))/batch,'hardware':hw.metrics(),'loss_with_untrained_head':lossvalue})
  print('BENCH',json.dumps(bench[-1]),flush=True)
 out={'status':'PASS','pretrained_checkpoint_sha256':hashlib.sha256((R/'input/deit_tiny_patch16_224-a1311bcf.pth').read_bytes()).hexdigest(),
  'independent_torchvision_max_logit_error':float((a-b).abs().max()),'pretrained_classes':1000,'CIFAR_head_classes':100,
  'signed_CIM_checks':checks,'all_fixed_weight_ops_W4A4':50,'pilot_hardware_targets':PILOT_TARGETS,
  'benchmark':bench,'test_data_used':False,'train_images':len(split['train']),'validation_images':len(split['validation']),
  'baseline_accuracy_not_yet_established':True,'hardware_assumption':{'ADC_bits':8,'raw_FS':192,'array':[128,128],'columns_per_read':128,'SA_window_LSB':8,'independent_synthetic_held_error_sigma_LSB':.03,'not_a_physical_fit':True}}
 save('preflight.json',out);print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
