from pathlib import Path
import sys,json,time,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
import torch.nn.functional as F
import run as runner
from models import build,attach,detach,prep,runtime,TARGETS
from hardware import Hardware
from deit_data import read_data
runtime();runner.compile_kernel()
x,y=read_data('train');ids=np.array(json.loads((R/'resnet/split.json').read_text())['validation'])
results={}
routes=[('digital',None,None,False),('native','adc',None,False),('disturbance_only','disturbance',None,True),('decision_only','sa',None,False),('full_direct','sa',None,True),('aware_full','sa','aware_20_skip4_best.pt',True),('aware_native','adc','aware_20_skip4_best.pt',False)]
for label,mode,checkpoint,dist in routes:
 m=build('resnet')
 if checkpoint:m.load_state_dict(torch.load(R/'resnet'/checkpoint,map_location='cpu',weights_only=True)['model'])
 hw=Hardware(mode,target=20,skip=4,disturbance=dist) if mode else None
 if hw:attach(m,hw,'resnet')
 layers=collections.defaultdict(list);hooks=[]
 for name in TARGETS['resnet']+['layer1.1.conv1','layer2.2.conv1','fc']:
  op=m.get_submodule(name)
  def hook(op,args,out,name=name):
   z=out.detach();qx,_,_,_=op.integers(args[0]);layers[name].append([float(z.mean()),float(z.std()),float((qx==0).float().mean()),float((qx==op.high).float().mean())])
  hooks.append(op.register_forward_hook(hook))
 zs=[];m.eval()
 with torch.inference_mode():
  for pos in range(0,len(ids),4):
   ix=ids[pos:pos+4]
   if hw:hw.context(ix,0)
   zs.append(m(prep(x[ix],'resnet')).cpu())
 z=torch.cat(zs);pred=z.argmax(1);cnt=collections.Counter(pred.tolist())
 results[label]=dict(accuracy=float((pred==y[ids]).float().mean()),CE=float(F.cross_entropy(z,y[ids])),predicted_classes=len(cnt),most_common_predictions=cnt.most_common(5),logit_std_over_samples=float(z.std(0).mean()),layers={n:np.mean(v,axis=0).tolist() for n,v in layers.items()},layer_fields=['output_mean','output_std','input_quantized_zero_fraction','input_quantized_at_upper_limit_fraction'],hardware=hw.metrics() if hw else None)
 for h in hooks:h.remove()
 (R/'diagnostics/resnet_failure.json').write_text(json.dumps(results,indent=2)+'\n')
 print(label,json.dumps({k:v for k,v in results[label].items() if k not in ('hardware','layers')}),flush=True)
 del m,hw;torch.cuda.empty_cache()
