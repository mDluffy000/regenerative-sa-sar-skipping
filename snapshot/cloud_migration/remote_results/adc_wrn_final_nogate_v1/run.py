"""Final predeclared no-extra-gating WRN matrix; no adaptive experiment expansion."""
import sys,os,json,time,copy,traceback
from pathlib import Path
import torch,numpy as np
from torch.utils.cpp_extension import load
import hardware as hwmod
from hardware import Hardware
from oracle import oracle
import train_wrn as t
R=Path(__file__).resolve().parent;P=hwmod.P;Q=Path('/root/autodl-tmp/adc_wrn28_10_extend_v1')
EPOCHS=20;TARGETS=[20,40];SKIPS=[4,5,6]

def report(state):
 t.RESULTS['status']=state;t.save('results.json',t.RESULTS)
 s='# Final WRN no-extra-zero-gating matrix\n\nStatus: '+state+'\n\n'
 s+='All valid mapped requests are converted; padding excluded structurally. Three mapped convolution layers, ADC8, 0–1.2V, fixed19/38mV detector thresholds. Training data2000/validation500; held table and interface assumptions unchanged.\n\n'
 s+='| Training stage | Finished epochs | Selected epoch | Best validation accuracy |\n|---|---:|---:|---:|\n'
 for k,v in t.RESULTS.get('stages',{}).items():s+=f"| {k} | {v['epochs_finished']} | {v['best_epoch']} | {100*v['validation']['accuracy']:.2f}% |\n"
 if t.RESULTS.get('test'):
  base=t.RESULTS['test']['native_ADC']['hardware']['comparisons']
  s+='\n| Test route | Accuracy | SA hit/all conversions | SAR comparison saving vs native | Whole conversion skipping | Comparisons |\n|---|---:|---:|---:|---:|---:|\n'
  for k,v in t.RESULTS['test'].items():
   h=v['hardware'];s+=f"| {k} | {100*v['accuracy']:.2f}% | {100*h['SA_hit_fraction_of_all_conversions']:.2f}% | {100*(1-h['comparisons']/base):.2f}% | 0% | {h['comparisons']} |\n"
 s+='\nSAR comparison counts are not total net energy or full-system latency. Circuit data: schematic only; sparse-grid extrapolation and single-ended985ps sampling are assumptions. Test set was observed in earlier project runs; no fresh blind-test claim. No extra gate hardware cost is assumed because data-dependent gates are disabled.\n'
 if t.RESULTS.get('reason'):s+='\n'+t.RESULTS['reason']
 (R/'REPORT.md').write_text(s)

def preflight():
 g=np.random.default_rng(914);raw=g.integers(0,50,(2,4,2,4,128)).astype(np.float32);raw[:,2]=0
 raw[...,::17]=g.integers(180,385,raw[...,::17].shape);raw[...,0:8]=[0,0,3,4,8,9,192,384]
 em=np.zeros((2,4,4),np.uint8);de=np.zeros((2,2,128),np.uint8);de[:,1,96:]=1
 ids=torch.tensor([7,9],device='cuda');args=[torch.from_numpy(v).cuda() for v in [raw,em,de]];checks=[]
 valid=2*4*4*(128+96);padding=raw.size-valid
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
 t.save('preflight.json',dict(status='PASS',checks=checks,events_checked=sum(v['events'] for v in checks),padding_only_exclusion=True,zero_samples_still_converted=True))

def main():
 with (R/'RUN_STARTED.json').open('x') as f:json.dump(dict(utc=t.now(),pid=os.getpid()),f)
 t.R=R;t.START=time.monotonic();t.runtime();t.report=report;t.Hardware=Hardware
 t.CFG.update(json.loads((Q/'protocol.json').read_text()));t.CFG.update(hardware_epochs=EPOCHS,hardware_lr=1e-5,maximum_wall_seconds=14400,held_sigma_LSB=0.,skip_bits=SKIPS,SA_window_LSB=None,zero_gating=False,hardware_assumption=json.loads((R/'model_contract.json').read_text()),final_protocol=json.loads((R/'FROZEN_PROTOCOL.json').read_text()))
 torch.manual_seed(t.CFG['seed']);t.RESULTS.clear();t.RESULTS.update(status='running',config=t.CFG,stages={});t.save('protocol.json',t.CFG)
 sources=list(R.glob('*.py'))+list(R.glob('*.cu'))+list(R.glob('*.bin'))+list((R/'input').iterdir())+[R/'model_contract.json',R/'FROZEN_PROTOCOL.json',Q/'W4A4_best.pt',Q/'pilot_split.json',Q/'test_ids.json',P/'calibration.json',P/'train_wrn.py',P/'gpu_hardware.py',P/'wrn.py',t.BASE/'model.py',t.BASE/'data.py']
 manifest={str(p):t.sha(p) for p in sources if p.is_file()};t.save('manifest.json',dict(sources=manifest,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
 (R/'build').mkdir(exist_ok=True)
 hwmod.EXT=load('wrn_final_nogate_v1',sources=[str(R/'events_cisa.cu')],build_directory=str(R/'build'),extra_cuda_cflags=['-O3','--fmad=false'],extra_cflags=['-O3'],verbose=False);preflight()
 if '--preflight-only' in sys.argv:raise RuntimeError('Use the standalone preflight command; main reserves final-run marker')
 x,y=t.read_data('train');pilot=json.loads((Q/'pilot_split.json').read_text());t.save('pilot_split.json',pilot);hids=np.array(pilot['hardware_train']);hval=np.array(pilot['hardware_validation']);assert not set(hids)&set(hval)
 q=t.quantize(t.WRN().cuda(),json.loads((P/'calibration.json').read_text())['scales']);q.load_state_dict(torch.load(Q/'W4A4_best.pt',weights_only=True)['model']);q.eval()
 t.RESULTS['initial_exact_check']=t.exact_check(q,x,hids);teacher=copy.deepcopy(q).eval()
 # Shared ordinary control receives equal epochs/images/teacher/loss, native ADC but no SA-specific perturbations.
 control=copy.deepcopy(q);hw=Hardware(mode='adc',disturbance=False);t.attach(control,hw)
 control=t.train('ordinary_ADC',control,x,y,hids,hval,EPOCHS,teacher,hw);t.RESULTS['ordinary_exact_check']=t.exact_check(control,x,hids);del control,hw;torch.cuda.empty_cache()
 for target in TARGETS:
  for skip in SKIPS:
   model=copy.deepcopy(q);hw=Hardware(target=target,skip=skip);t.attach(model,hw)
   name=f'aware_{target}_skip{skip}';model=t.train(name,model,x,y,hids,hval,EPOCHS,teacher,hw)
   t.RESULTS[name+'_exact_check']=t.exact_check(model,x,hids);del model,hw;torch.cuda.empty_cache()
 t.save('SELECTION_FROZEN.json',dict(utc=t.now(),stages=t.RESULTS['stages'],targets=TARGETS,skips=SKIPS))
 tx,ty=t.read_data('test');test=np.array(json.loads((Q/'test_ids.json').read_text()));t.save('test_ids.json',test.tolist());t.RESULTS['test']={}
 def evaluate(name,m,mode='sa',target=20,skip=6,dist=True):
  h=Hardware(mode,target,skip,dist);t.attach(m,h);v=t.evaluate(m,tx,ty,test,h,'test_'+name,t.CFG['test_perturbation_seed']);t.RESULTS['test'][name]=v
  z=v['hardware'];assert z['gated']==0 and z['conversions']==z['requests'];assert z['comparisons']==8*z['conversions']-skip*z['near']
  if name!='native_ADC':assert z['requests']==t.RESULTS['test']['native_ADC']['hardware']['requests']
  report('running')
 evaluate('native_ADC',q,'adc',dist=False)
 control=copy.deepcopy(q);control.load_state_dict(torch.load(R/'ordinary_ADC_best.pt',weights_only=True)['model']);evaluate('ordinary_native_ADC',control,'adc',dist=False)
 for target in TARGETS:
  evaluate(f'{target}_disturbance_only_direct',q,'disturbance',target,6,True)
  for skip in SKIPS:
   suffix=f'{target}_skip{skip}'
   evaluate('decision_only_'+suffix,q,'sa',target,skip,False)
   evaluate('direct_'+suffix,q,'sa',target,skip,True)
   evaluate('ordinary_'+suffix,control,'sa',target,skip,True)
   model=copy.deepcopy(q);model.load_state_dict(torch.load(R/f'aware_{suffix}_best.pt',weights_only=True)['model']);evaluate('aware_'+suffix,model,'sa',target,skip,True);del model;torch.cuda.empty_cache()
 t.RESULTS['paired']={}
 for target in TARGETS:
  for skip in SKIPS:
   suffix=f'{target}_skip{skip}';b=np.load(R/f'test_aware_{suffix}.npz');cb=b['logits'].argmax(1)==b['labels']
   for route in ['direct','ordinary']:
    a=np.load(R/f'test_{route}_{suffix}.npz');assert np.array_equal(a['ids'],b['ids']);ca=a['logits'].argmax(1)==a['labels'];t.RESULTS['paired'][f'{suffix}_vs_{route}']=dict(gain_pp=float((cb.astype(float)-ca).mean()*100),wrong_to_correct=int((~ca&cb).sum()),correct_to_wrong=int((ca&~cb).sum()))
 a0=np.load(Q/'test_native_ADC.npz');b0=np.load(R/'test_native_ADC.npz');assert np.array_equal(a0['ids'],b0['ids']) and np.array_equal(a0['logits'],b0['logits']);t.RESULTS['native_ADC_logits_invariant_to_zero_gate_removal']=True
 for p,h in manifest.items():assert t.sha(Path(p))==h,p
 t.RESULTS['elapsed_seconds']=time.monotonic()-t.START;report('complete');t.save('status.json',dict(state='complete',utc=t.now()))
 (R/'SHA256SUMS').write_text(''.join(t.sha(p)+'  '+p.name+'\n' for p in sorted(R.iterdir()) if p.is_file() and p.name not in ['SHA256SUMS','pipeline.log']))
if __name__=='__main__':
 if (R/'RUN_STARTED.json').exists():raise SystemExit('Existing run preserved')
 try:main()
 except Exception:
  t.RESULTS['reason']=traceback.format_exc();report('failed');t.save('status.json',dict(state='failed',utc=t.now(),reason=t.RESULTS['reason']));raise
