"""Exploratory WRN BN repair and matched no-gating sparsity ablation."""
import sys,os,json,time,copy,traceback,hashlib,tarfile
from pathlib import Path
R=Path('/root/autodl-tmp/adc_wrn_final_nogate_v1'); P=Path('/root/autodl-tmp/adc_wrn28_10_v1');Q=Path('/root/autodl-tmp/adc_wrn28_10_extend_v1')
O=Path('/root/autodl-tmp/adc_wrn_bn_repair_v1')
sys.path[:0]=[str(R),str(P)]
import torch,numpy as np
from torch import nn
from torch.utils.cpp_extension import load
import hardware as hm
from hardware import Hardware
import train_wrn as t
POINTS=[(20,4),(20,5),(20,6),(40,4),(40,5),(40,6)]
BRANCHES=[('native_control','adc',20,6,False),('prefix_only_20_skip6','sa',20,6,False),('voltage_only_20','disturbance',20,6,True)]+[(f'combined_{a}_skip{k}','sa',a,k,True) for a,k in [(20,5),(20,6),(40,4),(40,5)]]
def save(n,d):t.save(n,d)
def report(state):
 t.RESULTS['status']=state;save('results.json',t.RESULTS)
def metrics_summary(v):
 h=v['hardware'];s=h['sparsity_diagnostic']
 return dict(accuracy=v['accuracy'],potential_gate_fraction=s['potential_zero_gated_events_NOT_skipped']/h['requests'],activation_zero_fraction=s['activation_zeros']/s['activation_elements'],SAR_comparison_reduction=h['comparison_saving'])
@torch.no_grad()
def recalibrate(m,h,x,ids):
 m.eval();before={n:p.detach().clone() for n,p in m.named_parameters()}
 for b in m.modules():
  if isinstance(b,nn.BatchNorm2d):b.reset_running_stats();b.momentum=None;b.train()
 for j in range(0,len(ids),32):
  ix=ids[j:j+32];h.context(ix,0);m(t.prep(x[ix]))
 assert all(torch.equal(v,dict(m.named_parameters())[n]) for n,v in before.items())
 m.eval()
def evaluate(name,m,h,x,y,ids):
 t.attach(m,h);v=t.evaluate(m,x,y,ids,h,name)
 z=v['hardware'];assert z['gated']==0 and z['requests']==z['conversions'];assert z['comparisons']==8*z['conversions']-h.skip*z['near']
 return v

def main():
 O.mkdir(exist_ok=True);assert not (O/'RUN_STARTED.json').exists()
 t.R=O;t.START=time.monotonic();t.runtime();t.Hardware=Hardware;t.report=report
 t.CFG.update(json.loads((Q/'protocol.json').read_text()));t.CFG.update(hardware_epochs=20,hardware_lr=1e-5,hardware_batch=4,maximum_wall_seconds=21600,zero_gating=False,held_sigma_LSB=0.)
 t.RESULTS=dict(status='running',stages={},diagnostics={},test={},config=t.CFG)
 save('RUN_STARTED.json',dict(pid=os.getpid(),utc=t.now()))
 save('PROTOCOL.json',dict(branches=BRANCHES,epochs=20,train=2000,validation=500,test=1000,BN='train-only reset cumulative mean, batch32, no augmentation, weights unchanged; then frozen statistics during training',selection='best validation accuracy then CE, epoch0 included; all branches frozen before test',causal_scope='BN calibration and subsequent training effects recorded separately; within fixed simulator, not universal causality',gating='OFF; digital-known potential gates counted only',history='exploratory repair of previous failed matrix; previously observed test set'))
 files=[p for folder in [R,P,Q] for p in folder.iterdir() if p.is_file() and p.suffix in ['.py','.cu','.bin','.json','.pt']]
 manifest={str(p):t.sha(p) for p in files};save('SOURCE_MANIFEST.json',manifest)
 (R/'build').mkdir(exist_ok=True);hm.EXT=load('wrn_final_nogate_v1',sources=[str(R/'events_cisa.cu')],build_directory=str(R/'build'),extra_cuda_cflags=['-O3','--fmad=false'],extra_cflags=['-O3'],verbose=False)
 import run as parent_run
 parent_run.preflight()
 x,y=t.read_data('train');split=json.loads((Q/'pilot_split.json').read_text());ids=np.array(split['hardware_train']);val=np.array(split['hardware_validation']);assert len(ids)==2000 and len(val)==500 and not set(ids)&set(val)
 save('split.json',dict(train=ids.tolist(),validation=val.tolist()))
 q=t.quantize(t.WRN().cuda(),json.loads((P/'calibration.json').read_text())['scales']);q.load_state_dict(torch.load(Q/'W4A4_best.pt',weights_only=True)['model']);q.eval();teacher=copy.deepcopy(q).eval()
 save('exact_parent.json',t.exact_check(copy.deepcopy(q),x,ids))
 # Existing trained weights: isolate the BN-only effect on all six points.
 for a,k in POINTS:
  name=f'old_{a}_skip{k}';m=copy.deepcopy(q);m.load_state_dict(torch.load(R/f'aware_{a}_skip{k}_best.pt',weights_only=True)['model']);h=Hardware('sa',a,k,True)
  before=evaluate(name+'_before',m,h,x,y,val);recalibrate(m,h,x,ids);after=evaluate(name+'_BN',m,h,x,y,val)
  torch.save(dict(model=m.state_dict(),validation=after),O/(name+'_BN.pt'))
  t.RESULTS['diagnostics'][name]=dict(before=before,BN_only=after,parameters_unchanged=True);report('diagnosing')
  print('BN_RESULT',name,metrics_summary(before),metrics_summary(after),flush=True)
  del m,h;torch.cuda.empty_cache()
 # Fresh matched branches all start at the original parent; identical seeds/data/loss.
 for name,mode,a,k,dist in BRANCHES:
  torch.manual_seed(t.CFG['seed']);m=copy.deepcopy(q);h=Hardware(mode,a,k,dist)
  before=evaluate(name+'_parent',m,h,x,y,val);recalibrate(m,h,x,ids);after=evaluate(name+'_BN',m,h,x,y,val)
  torch.save(dict(model=m.state_dict(),validation=after),O/(name+'_BN.pt'))
  save(name+'_BN_effect.json',dict(before=before,after=after))
  m=t.train(name,m,x,y,ids,val,20,teacher,h)
  save(name+'_exact.json',t.exact_check(m,x,ids));del m,h;torch.cuda.empty_cache()
 save('SELECTION_FROZEN.json',dict(utc=t.now(),stages=t.RESULTS['stages']))
 tx,ty=t.read_data('test');test=np.array(json.loads((Q/'test_ids.json').read_text()));save('test_ids.json',test.tolist())
 for name,mode,a,k,dist in BRANCHES:
  for phase in ['BN','best']:
   m=copy.deepcopy(q);m.load_state_dict(torch.load(O/f'{name}_{phase}.pt',weights_only=True)['model']);h=Hardware(mode,a,k,dist)
   t.RESULTS['test'][name+'_'+phase]=evaluate('test_'+name+'_'+phase,m,h,tx,ty,test);report('testing');del m,h;torch.cuda.empty_cache()
 # All four ablation checkpoints evaluated under identical inference behaviors.
 t.RESULTS['ablation_validation']={}
 for name in ['native_control','prefix_only_20_skip6','voltage_only_20','combined_20_skip6']:
  m=copy.deepcopy(q);m.load_state_dict(torch.load(O/f'{name}_best.pt',weights_only=True)['model'])
  for behavior,mode,dist in [('native','adc',False),('prefix_only','sa',False),('voltage_only','disturbance',True),('combined','sa',True)]:
   h=Hardware(mode,20,6,dist);v=evaluate('cross_'+name+'_'+behavior,m,h,x,y,val);t.RESULTS['ablation_validation'][name+'__'+behavior]=v
  del m;torch.cuda.empty_cache();report('cross_evaluation')
 for p,sha in manifest.items():assert t.sha(Path(p))==sha,p
 report('complete');save('status.json',dict(state='complete',utc=t.now()))
 rows=['# WRN BN repair and sparsity ablation','', '| Route | Top-1 (%) | SAR reduction (%) | Potential zero-gate (%) |','|---|---:|---:|---:|']
 for n,v in t.RESULTS['test'].items():
  z=metrics_summary(v);rows.append(f"| {n} | {100*z['accuracy']:.2f} | {100*z['SAR_comparison_reduction']:.4f} | {100*z['potential_gate_fraction']:.4f} |")
 (O/'REPORT.md').write_text('\n'.join(rows)+'\n\nGating OFF throughout. Potential opportunities are digital-known masks, not free analog-zero detection. Exploratory single-seed result; errors/BN/calibration scopes retained in protocol.\n')
 save('SHA256.json',{p.name:t.sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['pipeline.log','SHA256.json']})
 with tarfile.open(O.parent/'wrn_bn_repair_v1_delivery.tar.gz','w:gz') as f:
  for p in O.iterdir():
   if p.is_file():f.add(p,arcname=p.name)
 print('COMPLETE',t.now(),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  O.mkdir(exist_ok=True);(O/'FAILURE.json').write_text(json.dumps(dict(traceback=traceback.format_exc())));raise
