import hashlib
from pathlib import Path
import numpy as np,json,re,math,yaml
from isaac_trace_reader import iter_events
ROOT=Path(__file__).resolve().parent

def main():
 z=np.load(ROOT/'isaac_w4a4/isaac_adc_pre_trace.npz');cost=json.loads((ROOT/'isaac_w4a4/cost_results.json').read_text());checks=[]
 for c in cost:
  p=c['layer'].replace('.','_')+'__';r=z[p+'raw'].astype(np.int64);sh=z[p+'meta'][:,3];ex=(r*sh[:,None,None,None]).sum(0)+z[p+'offset_corrections'].sum(0)
  np.testing.assert_array_equal(ex,z[p+'input_q'].astype(np.int64)@z[p+'weight_q'].astype(np.int64).T)
  np.testing.assert_array_equal(np.rint(np.clip(r,0,192)*255/192).astype(np.uint8),z[p+'code'])
  ad=(z[p+'code'].astype(np.float64)*(192/255)*sh[:,None,None,None]).sum(0)+z[p+'offset_corrections'].sum(0)
  np.testing.assert_allclose(ad,z[p+'adc'],atol=1e-9,rtol=0)
  folder=ROOT/'isaac_w4a4'/('cost_compat1_'+c['layer'].replace('.','_'));stats=(folder/'timeloop-mapper.stats.txt').read_text().split('=== adc ===')[1].split('Level ')[0]
  val=lambda k:float(re.search(re.escape(k)+r'\s*:\s*([\d.eE+-]+)',stats).group(1))
  action=val('Scalar reads (per-instance)')*val('Utilized instances (max)')/val('Block size')
  assert action==r.size==c['adc_event_requests']
  ert=yaml.safe_load((folder/'timeloop-mapper.ERT.yaml').read_text())
  t=next(t for t in ert['ERT']['tables'] if '.adc[' in t['name']);e=next(a['energy'] for a in t['actions'] if a['name']=='read')
  assert math.isclose(action*e*1e-12,c['per_component_energy']['adc'],rel_tol=.006)
  checks.append(dict(layer=c['layer'],trace_events=r.size,timeloop_adc_read_actions=action,adc_read_energy_pj=e,energy_count_check_relative_error=abs(action*e*1e-12/c['per_component_energy']['adc']-1)))
 # Validate per-lane service ordering, read grouping and scalar normalized interface.
 order_hash=hashlib.sha256();last={};count=0
 for e in iter_events():
  order_hash.update(repr((e['layer'],e['sample_id'],e['output_window'],e['array_id'],e['adc_lane_id'],e['read_id'],e['raw_psum'],e['adc_code'])).encode())
  key=(e['layer'],e['array_id']);prev=last.get(key)
  if prev:
   assert e['lane_sequence']==prev['lane_sequence']+1
   assert e['service_time_ns']>prev['service_time_ns']
   if e['read_id']==prev['read_id']:assert abs(e['d_lsb_previous_same_read']-(e['raw_psum']-prev['raw_psum'])/e['adc_lsb_raw'])<1e-10
  last[key]=e;count+=1
 assert count==sum(c['trace_events'] for c in checks)
 result=dict(ordered_stream_sha256=order_hash.hexdigest(),status='pass_with_declared_transfer_and_timing_supplements',public_validation='official test_energy_breakdown with CIM_ARCHITECTURE=True compatibility injection',integer_exact=True,trace_replay=True,service_order_verified=True,trace_events=count,checks=checks,limitations=['public model has no per-sample ADC transfer; linear lower-half curve is explicit supplement','public timing lacks ADC throughput constraint; 100ns read envelope is paper-derived schedule, not public cycles'])
 (ROOT/'isaac_acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
