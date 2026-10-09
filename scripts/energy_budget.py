"""Recompute the paper's reference-calibrated estimate, not a silicon measurement."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[1];out=R/'generated';out.mkdir(exist_ok=True)
rows=[]
for target,k,cmp_pct,detector_fj in [(20,4,27.30,57.10),(40,6,48.19,61.31)]:
    reference_pj=1.929
    gross_pct=.63*cmp_pct
    residual_pj=reference_pj*gross_pct/100-detector_fj/1000
    rows.append(dict(target_mV=target,skip=k,comparison_reduction_pct=cmp_pct,
        reference_ADC_pJ=reference_pj,reference_cycle_fraction=.63,detector_fJ=detector_fj,
        gross_reduction_pct=gross_pct,net_reduction_pct=residual_pj/reference_pj*100,
        residual_pJ_per_conversion=residual_pj))
with (out/'energy_budget.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
