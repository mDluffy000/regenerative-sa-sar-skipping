"""Run official CiMLoop validation entry, sequentially, into a fresh result folder."""
import sys,json,argparse,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PUB=ROOT/'cimloop_public/workspace'
sys.path.insert(0,str(PUB))
from scripts import utils as u

def main():
    ap=argparse.ArgumentParser();ap.add_argument('macro');ap.add_argument('output');args=ap.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    u.get_run_dir=lambda:str(out)
    original_get_spec=u.get_spec
    def compatible_get_spec(*a,**kw):
        spec=original_get_spec(*a,**kw)
        spec.variables.setdefault('CIM_ARCHITECTURE',True)
        return spec
    u.get_spec=compatible_get_spec
    original_parallel=u.parallel_test
    u.parallel_test=lambda calls:original_parallel(calls,n_jobs=1)
    result=u.get_test(args.macro,'test_energy_breakdown')()
    rows=[]
    for r in result:
        rows.append({k:getattr(r,k) for k in ('computes','cycles','cycle_seconds','energy','area','per_component_energy','per_component_area','variables','mapping')})
    (out/'public_summary.json').write_text(json.dumps(rows,indent=2,default=str))
    print(json.dumps([{k:r[k] for k in ('computes','cycles','cycle_seconds','energy','area')} for r in rows],indent=2))
if __name__=='__main__':main()
