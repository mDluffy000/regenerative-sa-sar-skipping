"""Train-only BN recalibration diagnostic; no weights/scales changed, no test reads."""
from pathlib import Path
import sys,json,time,copy,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
from torch import nn
import torch.nn.functional as F
import run as runner
from models import build,attach,prep,runtime
from hardware import Hardware
from deit_data import read_data
runtime();runner.compile_kernel();torch.manual_seed(20260913)
O=R/'diagnostics/bn_refresh';O.mkdir(exist_ok=False)
x,y=read_data('train');s=json.loads((R/'resnet/split.json').read_text());train=np.array(s['train']);valid=np.array(s['validation']);assert not set(train)&set(valid)
protocol=dict(routes=['native_ADC','SA20_skip4','SA40_skip6'],initialization='same original digital W4A4 checkpoint for all three',calibration='one pass 2000 train images, batch32, no augmentation, reset BN running mean/var and cumulative average, no optimizer',labels_used=False,validation_images=500,test_used=False,selection='no selection; report all three routes',SA_model='unchanged from WRN final no-gate')
(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
result={}
for name,target,skip in [('native_ADC',None,4),('SA20_skip4',20,4),('SA40_skip6',40,6)]:
 m=build('resnet');h=Hardware('adc',disturbance=False) if target is None else Hardware(target=target,skip=skip)
 attach(m,h,'resnet');params={n:p.detach().clone() for n,p in m.named_parameters()};m.eval();zs=[]
 with torch.inference_mode():
  for pos in range(0,len(valid),4):
   ix=valid[pos:pos+4];h.context(ix,0);zs.append(m(prep(x[ix],'resnet')).cpu())
 z=torch.cat(zs);before=float((z.argmax(1)==y[valid]).float().mean())
 for op in m.modules():
  if isinstance(op,nn.BatchNorm2d):op.reset_running_stats();op.momentum=None;op.train()
 with torch.no_grad():
  for pos in range(0,len(train),32):
   ix=train[pos:pos+32];h.context(ix,0);m(prep(x[ix],'resnet'))
 assert all(torch.equal(v,dict(m.named_parameters())[n]) for n,v in params.items())
 m.eval();h.reset_stats();zs=[];sat=[]
 def hook(op,args):
  q=op.integers(args[0])[0];sat.append(float((q==15).float().mean()))
 hook_handle=m.fc.register_forward_pre_hook(hook)
 with torch.inference_mode():
  for pos in range(0,len(valid),4):
   ix=valid[pos:pos+4];h.context(ix,0);zs.append(m(prep(x[ix],'resnet')).cpu())
 hook_handle.remove();z=torch.cat(zs);pred=z.argmax(1)
 result[name]=dict(before_accuracy=before,after_accuracy=float((pred==y[valid]).float().mean()),CE=float(F.cross_entropy(z,y[valid])),predicted_classes=len(set(pred.tolist())),final_input_saturation=float(np.mean(sat)),hardware=h.metrics(),all_trainable_parameters_unchanged=True)
 torch.save(dict(model=m.state_dict(),route=name,calibration=protocol),O/(name+'.pt'))
 np.savez_compressed(O/(name+'.npz'),ids=valid,labels=y[valid].numpy(),logits=z.numpy())
 (O/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(name,json.dumps({k:v for k,v in result[name].items() if k!='hardware'}),flush=True)
 del m,h;torch.cuda.empty_cache()
