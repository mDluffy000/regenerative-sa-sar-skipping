"""Predeclared 2x2 training ablation of voltage disturbance and prefix reuse."""
import sys,os,json,time,copy,traceback
from pathlib import Path
import torch,numpy as np
from torch.utils.cpp_extension import load
import hardware as hwmod
from hardware import Hardware
from oracle import oracle
import train_wrn as t
R=Path(__file__).resolve().parent;P=hwmod.P;Q=Path('/root/autodl-tmp/adc_wrn28_10_extend_v1');FINAL=Path('/root/autodl-tmp/adc_wrn_final_nogate_v1')
TARGETS=[20];SKIPS=[6];EPOCHS=20
BRANCHES=[('ordinary','adc',False),('voltage_only','disturbance',True),('prefix_only','sa',False),('combined','sa',True)]

def report(state):
 t.RESULTS['status']=state;t.save('results.json',t.RESULTS)
 s='# Sparsity mechanism matched-training ablation\n\nStatus: '+state+'\n\nAll4 branches start from original W4A4,20epochs each, same data/seeds/teacher/loss. Target20mV, skip6. Data-dependent gating ON only to diagnose prior experiments. No new test-set evaluation.\n\n'
 s+='| Branch | Finished epochs | Selected epoch | Best validation accuracy |\n|---|---:|---:|---:|\n'
 for k,v in t.RESULTS.get('stages',{}).items():s+=f"| {k} | {v['epochs_finished']} | {v['best_epoch']} | {100*v['validation']['accuracy']:.2f}% |\n"
 if t.RESULTS.get('evaluation'):
  s+='\n| Weights / inference behavior | Validation accuracy | Gated events | Requests | Conversions |\n|---|---:|---:|---:|---:|\n'
  for k,v in t.RESULTS['evaluation'].items():
   h=v['hardware'];s+=f"| {k} | {100*v['accuracy']:.2f}% | {h['gated']} | {h['requests']} | {h['conversions']} |\n"
 s+='\nMatched factors identify effects under this frozen simulator, not a universal causal claim or validated physical ADC interface. Per-layer masks and quantization-scale summaries are saved. No free zero-detector energy claim.\n'
 if t.RESULTS.get('reason'):s+='\n'+t.RESULTS['reason']
 (R/'REPORT.md').write_text(s)

def weight_stats(m):
 out={}
 for n,op in m.named_modules():
  if isinstance(op,t.QuantOp):
   w=op.weight.detach();sc=op.weight_scale.detach().clamp(min=1e-8);q=torch.round((w/sc).clamp(-7,7));a=op.activation_scale.detach()
   out[n]={'quantized_weight_zero_fraction':float((q==0).float().mean()),'weight_scale_mean':float(sc.mean()),'weight_scale_min':float(sc.min()),'weight_scale_max':float(sc.max()),'activation_scale':a.cpu().reshape(-1).tolist(),'weight_L2':float(w.norm())}
 return out

def main():
 assert json.loads((FINAL/'status.json').read_text())['state']=='complete'
 assert (R/'PHASE1_ARCHIVED.json').exists(),'Main matrix must be archived before cause training'
 with (R/'RUN_STARTED.json').open('x') as f:json.dump(dict(utc=t.now(),pid=os.getpid()),f)
 t.R=R;t.START=time.monotonic();t.runtime();t.report=report;t.Hardware=Hardware
 t.CFG.update(json.loads((Q/'protocol.json').read_text()));t.CFG.update(hardware_epochs=20,hardware_lr=1e-5,maximum_wall_seconds=10800,skip_bits=6,held_sigma_LSB=0.,SA_window_LSB=None,zero_gating=True,cause_protocol=json.loads((R/'FROZEN_PROTOCOL.json').read_text()))
 torch.manual_seed(t.CFG['seed']);t.RESULTS.clear();t.RESULTS.update(status='running',config=t.CFG,stages={});t.save('protocol.json',t.CFG)
 sources=list(R.glob('*.py'))+list(R.glob('*.cu'))+list(R.glob('*.bin'))+list((R/'input').iterdir())+[R/'FROZEN_PROTOCOL.json',Q/'W4A4_best.pt',Q/'pilot_split.json',P/'calibration.json',P/'train_wrn.py',P/'gpu_hardware.py',P/'wrn.py',t.BASE/'model.py',t.BASE/'data.py']
 manifest={str(p):t.sha(p) for p in sources if p.is_file()};t.save('manifest.json',dict(sources=manifest,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
 (R/'build').mkdir(exist_ok=True);hwmod.EXT=load('wrn_sparsity_cause_v1',sources=[str(R/'events_cisa.cu')],build_directory=str(R/'build'),extra_cuda_cflags=['-O3','--fmad=false'],extra_cflags=['-O3'],verbose=False)
 preflight()
 x,y=t.read_data('train');pilot=json.loads((Q/'pilot_split.json').read_text());t.save('pilot_split.json',pilot);hids=np.array(pilot['hardware_train']);hval=np.array(pilot['hardware_validation']);assert not set(hids)&set(hval)
 q=t.quantize(t.WRN().cuda(),json.loads((P/'calibration.json').read_text())['scales']);q.load_state_dict(torch.load(Q/'W4A4_best.pt',weights_only=True)['model']);q.eval();t.RESULTS['initial_exact_check']=t.exact_check(q,x,hids);teacher=copy.deepcopy(q).eval()
 t.save('original_weight_stats.json',weight_stats(q))
 for name,mode,dist in BRANCHES:
  m=copy.deepcopy(q);h=Hardware(mode,20,6,dist);t.attach(m,h);m=t.train(name,m,x,y,hids,hval,20,teacher,h)
  t.RESULTS[name+'_exact_check']=t.exact_check(m,x,hids);t.save(name+'_weight_stats.json',weight_stats(m));del m,h;torch.cuda.empty_cache()
 t.save('SELECTION_FROZEN.json',dict(utc=t.now(),stages=t.RESULTS['stages']));t.RESULTS['evaluation']={}
 for source in ['original']+[b[0] for b in BRANCHES]:
  m=copy.deepcopy(q)
  if source!='original':m.load_state_dict(torch.load(R/f'{source}_best.pt',weights_only=True)['model'])
  for behavior,mode,dist in BRANCHES:
   h=Hardware(mode,20,6,dist);t.attach(m,h);key=f'{source}__{behavior}';v=t.evaluate(m,x,y,hval,h,'validation_'+key,t.CFG['validation_perturbation_seed']);t.RESULTS['evaluation'][key]=v
   assert v['hardware']['sparsity_diagnostic']['zero_gated_events']==v['hardware']['gated'];report('running')
  del m,h;torch.cuda.empty_cache()
 for p,h in manifest.items():assert t.sha(Path(p))==h,p
 t.RESULTS['elapsed_seconds']=time.monotonic()-t.START;report('complete');t.save('status.json',dict(state='complete',utc=t.now()))
 (R/'SHA256SUMS').write_text(''.join(t.sha(p)+'  '+p.name+'\n' for p in sorted(R.iterdir()) if p.is_file() and p.name not in ['SHA256SUMS','pipeline.log']))

def preflight():
 g=np.random.default_rng(914);raw=g.integers(0,50,(2,4,2,4,128)).astype(np.float32);raw[:,2]=0
 raw[...,::17]=g.integers(180,385,raw[...,::17].shape);raw[...,0:8]=[0,0,3,4,8,9,192,384]
 em=(g.random((2,4,4))<.15).astype(np.uint8);de=(g.random((2,2,128))<.12).astype(np.uint8);de[:,1,96:]=1
 ids=torch.tensor([7,9],device='cuda');args=[torch.from_numpy(v).cuda() for v in [raw,em,de]];checks=[]
 excluded=em[:,:,None,:,None].astype(bool)|de[:,None,:,None,:].astype(bool);padding=int(excluded.sum());valid=raw.size-padding
 for target in TARGETS:
  for skip in SKIPS:
   for mode,dist in [(0,False),(1,False),(2,True),(3,False),(3,True)]:
    h=Hardware(target=target,skip=skip,disturbance=dist)
    out,s,tr=hwmod.EXT.convert(*args,ids,2,3,0,mode,skip,h.theta,0.,True,h.table,dist,h.lower,h.upper)
    aa,ss,tt=oracle(raw,em,de,mode,h.table.cpu().numpy(),dist,h.theta,h.lower,h.upper,skip)
    assert np.array_equal(out.cpu().numpy(),aa) and np.array_equal(s.cpu().numpy(),ss),(target,skip,mode,'codes/counts')
    np.testing.assert_allclose(tr.cpu().numpy(),tt,rtol=0,atol=1e-11,equal_nan=True)
    assert ss[1]==padding
    if mode:assert ss[2]==valid and ss[5]==8*valid-skip*ss[4]
    checks.append(dict(target=target,skip=skip,mode=mode,disturbance=dist,events=raw.size))
 t.save('preflight.json',dict(status='PASS',checks=checks,events_checked=sum(v['events'] for v in checks),CPU_GPU_gating_masks_match=True))


if __name__=='__main__':
 if (R/'RUN_STARTED.json').exists():raise SystemExit('Existing run preserved')
 try:main()
 except Exception:
  t.RESULTS['reason']=traceback.format_exc();report('failed');t.save('status.json',dict(state='failed',utc=t.now(),reason=t.RESULTS['reason']));raise
