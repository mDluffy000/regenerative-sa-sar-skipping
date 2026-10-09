"""Serialize executed ISAAC read order; parallel lanes share timestamps."""
from pathlib import Path
import numpy as np,json,csv,sys,itertools,argparse
from mapped_order import mapped_windows
ROOT=Path(__file__).resolve().parent

def iter_events():
 z=np.load(ROOT/'isaac_w4a4/isaac_adc_pre_trace.npz')
 for macro,d in enumerate(json.loads((ROOT/'isaac_w4a4/trace_index.json').read_text())):
  pre=d['prefix']+'__';raw=z[pre+'raw'];code=z[pre+'code'];meta=z[pre+'meta'];G,N,P,C=raw.shape;chunks=G//8;sequence=0;step=d['lsb_raw']
  for ordinal,(n,p) in enumerate(mapped_windows('isaac',d['prefix'],N,P)):
    for bit in range(4):
     read=ordinal*4+bit
     for col in range(2*C):
      sl,c=divmod(col,C)
      for array in range(chunks):
       g=array*8+bit*2+sl;value=int(raw[g,n,p,c]);previous=None
       if col:
        ps,pc=divmod(col-1,C);previous=int(raw[array*8+bit*2+ps,n,p,pc])
       yield dict(architecture='isaac_isca_2016',workload='ResNet20_CIFAR100_W4A4',sample_id=n,output_window=p,layer=d['layer'],tile_id=0,macro_id=macro,array_id=array,bank_id=array,adc_lane_id=array,event_sequence=sequence,lane_sequence=read*(2*C)+col,read_id=read,activation_bit=bit,weight_slice=sl,column_id=col,output_channel=c,sign_encoding='offset',weight_offset=7,activation_offset=0,reconstruction_shift=1<<(bit+2*sl),candidate_scale=float(z[pre+'candidate_scale'][c]),raw_psum=value,analog_normalized_fs=value/d['adc_fs_raw'],voltage_v=None,adc_full_scale_raw=d['adc_fs_raw'],adc_lsb_raw=step,value_lsb=value/step,d_lsb_previous_same_read=(value-previous)/step if previous is not None else None,adc_code=int(code[g,n,p,c]),service_time_ns=read*100+col/1.28,read_period_ns=100,physical_voltage_status='unverified',ordering_status='defined_serial_schedule_supplement')
       sequence+=1
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20);n=ap.parse_args().limit
 gen=iter_events();gen=itertools.islice(gen,n) if n else gen;w=None
 for r in gen:
  if w is None:w=csv.DictWriter(sys.stdout,fieldnames=list(r));w.writeheader()
  w.writerow(r)
