from pathlib import Path
import json,csv,re,statistics
from window_analyze import read,write
from held_analyze import extract
R=Path(__file__).resolve().parent
names=json.loads((R/'jobs_wr2full.json').read_text());rs=sum([read(n)for n in names],[]);assert len(rs)==732 and all(r['pass']for r in rs)
summary=[];groups=[]
for held in[False,True]:
 for t in[20,40]:
  xx=[r for r in rs if ('_held_'in r['job'])==held and r['target_mV']==t];cms=sorted(set(r['cm_V']for r in xx))
  for cm in cms:
   x=[r for r in xx if r['cm_V']==cm];groups.append(dict(held_50fF=held,target_mV=t,cm_V=cm,N=len(x),complete_guard_set=len(x)==5,pass_count=sum(r['pass']for r in x),subtotal_min_fJ=min(r['subtotal_draw_fJ']for r in x),subtotal_max_fJ=max(r['subtotal_draw_fJ']for r in x),stable_max_ps=max(r['stable_service_ps']for r in x)))
  complete=[g['cm_V']for g in groups if g['held_50fF']==held and g['target_mV']==t and g['complete_guard_set']]
  summary.append(dict(held_50fF=held,target_mV=t,N=len(xx),pass_count=sum(r['pass']for r in xx),cm_points=len(cms),full_guard_cm_points=len(complete),full_guard_min_cm=min(complete),full_guard_max_cm=max(complete),subtotal_min_fJ=min(r['subtotal_draw_fJ']for r in xx),subtotal_median_fJ=statistics.median(r['subtotal_draw_fJ']for r in xx),subtotal_max_fJ=max(r['subtotal_draw_fJ']for r in xx),all_ports_max_fJ=max(r['all_ports_draw_fJ']for r in xx),stable_max_ps=max(r['stable_service_ps']for r in xx)))
write('final_events.csv',rs);write('final_by_cm.csv',groups);write('final_summary.csv',summary)
held=sum([extract(f'w_r2_held_{t}',f'w_r2_removed_{t}')for t in[20,40]],[]);write('held_disturbance_vs_removed.csv',held)
# Baseline-subtracted acquisition source positive deliveries (diagnostic, ideal external drivers).
added=[]
for t in[20,40]:
 s=(R/f'w_r2_removed_{t}'/'spectre.out').read_text();en={(int(m[1]),int(m[2])):float(m[3].split(',')[1])*1e15 for m in re.finditer(r'ENERGY,(\d+),(\d+),([^\n]+)',s)}
 for r in read(f'w_r2_held_{t}'):
  p=en[r['slot'],4];n=en[r['slot'],5]
  added.append(dict(target_mV=t,slot=r['slot'],cm_V=r['cm_V'],d_mV=r['d_mV'],detector_subtotal_fJ=r['subtotal_draw_fJ'],input_acquisition_draw_fJ=r['input_p_draw_fJ']+r['input_n_draw_fJ'],removed_acquisition_draw_fJ=p+n,baseline_adjusted_measured_draw_fJ=r['all_ports_draw_fJ']-p-n))
write('baseline_adjusted_energy_exploratory.csv',added)
print(json.dumps(summary,indent=2));print('Baseline adjusted range',min(r['baseline_adjusted_measured_draw_fJ']for r in added),max(r['baseline_adjusted_measured_draw_fJ']for r in added))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
for j,t in enumerate([20,40]):
 for heldv,c,label in [(False,'#4a77b6','Ideal driven inputs'),(True,'#cd772e','50 fF held inputs')]:
  x=[g for g in groups if g['target_mV']==t and g['held_50fF']==heldv]
  axs[0,j].plot([g['cm_V']for g in x],[g['subtotal_max_fJ']for g in x],'.-',color=c,label=label,lw=1,ms=4)
  x=[g for g in x if g['complete_guard_set']]
  axs[1,j].plot([g['cm_V']for g in x],[g['pass_count']for g in x],'.-',color=c,label=label,lw=1,ms=5)
 axs[0,j].axhline(80,color='0.5',ls='--',label='80 fJ stopping budget');axs[0,j].set_ylim(25,85);axs[0,j].set_title(f'{t} mV class / fixed ratio {t*.95:g} mV')
 axs[1,j].set_ylim(0,5.5);axs[1,j].set_xlabel('External pair common mode (V)');axs[1,j].set_yticks([0,1,2,3,4,5])
 for ax in axs[:,j]:ax.grid(alpha=.2)
axs[0,0].set_ylabel('Max measured subtotal / event (fJ)');axs[1,0].set_ylabel('Passed guards per complete CM set');axs[0,0].legend(fontsize=8)
fig.suptitle('TT schematic validation: same settings across common mode\n732/732 legal events pass; rail endpoints only support zero-difference tests',fontsize=12)
fig.tight_layout();fig.savefig(R/'wide_cm_final.png',dpi=170)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for t,c in[(20,'#4a77b6'),(40,'#cd772e')]:
 x=[r for r in held if r['job'].endswith(str(t))]
 for i,mode in enumerate(['CM','DIFF']):axs[i].scatter([r['cm_V']for r in x],[r[f'HADC_{mode}_mV']for r in x],s=25,label=f'{t} mV class',alpha=.65,color=c)
for ax,mode in zip(axs,['Common mode','Differential']):ax.set_xlabel('External common mode (V)');ax.set_ylabel('Detector minus removed baseline (mV)');ax.set_title(mode);ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('50 fF hold residual at event phase 985 ps (timing sensitivity; not frozen ADC timing)');fig.tight_layout();fig.savefig(R/'held_residual_final.png',dpi=170)
