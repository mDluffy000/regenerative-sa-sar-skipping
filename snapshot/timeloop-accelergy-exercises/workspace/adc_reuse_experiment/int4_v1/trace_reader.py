"""Materialize the broadcast/axis trace schema as event records, lazily."""
import json,argparse,itertools,csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def iter_events(root=ROOT):
    with np.load(root/'adc_trace_sample.npz') as z:
        for d in json.loads((root/'trace_index.json').read_text()):
            p=d['prefix']+'__';raw=z[p+'raw_integer_psum'];code=z[p+'adc_code'];norm=z[p+'analog_normalized_fs'];ev=z[p+'zero_evidence_type'];gcount,ncount,windows,columns=raw.shape
            for g,n,w,c in np.ndindex(raw.shape):
                read=(g*ncount+n)*windows+w
                yield dict(run_id=str(z[p+'run_id']),sample_id=int(z[p+'sample_id'][n]),layer_id=d['layer'],adc_lane_id=0,bank_id=0,read_id=read,event_order=read*columns+c,column_id=c,row_chunk_id=int(z[p+'row_chunk_id'][g]),activation_bit=int(z[p+'activation_bit'][g]),weight_slice=int(z[p+'weight_slice'][g]),sign_branch=int(z[p+'sign_branch'][g]),raw_integer_psum=int(raw[g,n,w,c]),analog_normalized_fs=float(norm[g,n,w,c]),analog_voltage_v=None,adc_code=int(code[g,n,w,c]),transfer_id=str(z[p+'transfer_id']),vfs_v=None,voltage_mapping_id=str(z[p+'voltage_mapping_id']),candidate_shift=int(z[p+'candidate_shift'][g]),candidate_scale=float(z[p+'candidate_scale'][c]),zero_preknown=bool(ev[g,n,w,c]),zero_evidence_type=int(ev[g,n,w,c]),physical_voltage_mapping_status='unverified')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20,help='0 streams every event');args=ap.parse_args()
    records=iter_events();records=itertools.islice(records,args.limit) if args.limit else records
    writer=None
    for r in records:
        if writer is None:writer=csv.DictWriter(sys.stdout,fieldnames=list(r));writer.writeheader()
        writer.writerow(r)
