"""Measured W4A4 data statistics and explicit one-macro placement; native ADC."""
import sys,json,math,re,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PUB=ROOT/'cimloop_public/workspace';sys.path.insert(0,str(PUB))
from scripts import utils as u

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',default='');tag=ap.parse_args().tag
 z=np.load(ROOT/'isaac_w4a4/isaac_adc_pre_trace.npz');index=json.loads((ROOT/'isaac_w4a4/trace_index.json').read_text());allrows=[]
 for d in index:
  prefix=d['prefix'];x=z[prefix+'__input_q'].astype(np.int64);w=z[prefix+'__weight_q'].astype(np.int64);N,P,R=x.shape;M=w.shape[0];chunks=math.ceil(R/128)
  path=ROOT/'configs'/f'isaac_{tag}_{prefix}_w4a4.yaml'
  path.write_text("{{include_text('"+str(PUB/'models/workloads/problem_base.yaml')+"')}}\nproblem:\n  <<<: *problem_base\n  name: "+prefix+"\n  instance:\n    N: 2\n    P: "+str(P)+"\n    C: "+str(chunks*128)+"\n    M: "+str(M)+"\n")
  out=ROOT/'isaac_w4a4'/f'cost_{tag}_{prefix}';out.mkdir(exist_ok=False);u.get_run_dir=lambda:str(out)
  spec=u.get_spec('isaac_isca_2016',layer=str(path),system='ws_dummy_buffer_one_macro')
  xp=np.pad(x,((0,0),(0,0),(0,chunks*128-R)));wp=np.pad(w+7,((0,0),(0,chunks*128-R)))
  ib=[float(((xp>>b)&1).mean()) for b in range(3,-1,-1)];wb=[float(((wp>>b)&1).mean()) for b in range(3,-1,-1)]
  vals={'CIM_ARCHITECTURE':True,'INPUT_BITS':4,'WEIGHT_BITS':4,'OUTPUT_BITS':32,'BATCH_SIZE':2,'AVERAGE_INPUT_VALUE':float(np.mean(ib)),'AVERAGE_WEIGHT_VALUE':float(np.mean([((wp>>(2*s))&3).mean()/3 for s in range(2)])),'INPUT_BIT_DISTRIBUTION':ib,'WEIGHT_BIT_DISTRIBUTION':wb}
  spec.variables.update(vals)
  for k in list(spec.variables):
   if k not in vals:spec.variables[k]=spec.variables.pop(k)
  for node,factors in [('array',[f'C={chunks}']),('column',['Y=2',f'M={M}']),('row',['C=128'])]:
   c=spec.architecture.name2leaf(node).constraints.spatial;c['factors_only']=u.tl.constraints.Factors(factors);c['maximize_dims'].clear()
  r=u.run_mapper(spec)
  record={k:getattr(r,k) for k in ['computes','cycles','cycle_seconds','energy','area','per_component_energy','per_component_area','variables','mapping']}
  record.update(layer=d['layer'],scope='one ISAAC macro, native 8 arrays, unused instances retained; no tile/router energy',output_accumulator_cost_bits=32,logical_macs=N*P*R*M,padded_macs=N*P*chunks*128*M,adc_event_requests=N*P*chunks*4*2*M,adc_service_lower_bound_seconds=N*P*4*(2*M)/1.28e9,read_cycle_100ns_envelope_seconds=N*P*4*100e-9,input_stats_source='actual adc8 trace operand bits, zero padded inactive rows',native_adc_resolution=8)
  (out/'summary.json').write_text(json.dumps(record,indent=2,default=str));allrows.append(record)
  print(d['layer'],{k:record[k] for k in ['energy','cycles','area','adc_event_requests']},flush=True)
 (ROOT/'isaac_w4a4/cost_results.json').write_text(json.dumps(allrows,indent=2,default=str))
if __name__=='__main__':main()
