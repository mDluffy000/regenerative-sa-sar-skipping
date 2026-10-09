from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent;R=H.parent
plt.style.use(H/'styles/iscas.mplstyle')
B=R/'cloud_migration/remote_results'
p=B/'adc_wrn28_10_extend_v1'
a=np.load(p/'test_floating.npz');b=np.load(p/'test_W4A4.npz')
assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels'])
bases={'FP32':100*(a['logits'].argmax(1)==a['labels']).mean(),'W4A4':100*(b['logits'].argmax(1)==b['labels']).mean()}
old=json.loads((B/'adc_wrn_final_nogate_v1/results.json').read_text());new=json.loads((B/'adc_wrn_bn_repair_v1/results.json').read_text())
rows=[]
for t in [20,40]:
 for k in [4,5,6]:
  prior=(t,k) in [(20,4),(40,6)]
  name=f'aware_{t}_skip{k}' if prior else f'combined_{t}_skip{k}_best'
  source=B/('adc_wrn_final_nogate_v1' if prior else 'adc_wrn_bn_repair_v1')
  v=(old if prior else new)['test'][name]
  if (t,k)==(40,6):
   source=B/'adc_wrn40_skip6_paired_v1'
   name='combined_40_skip6_best'
   v=json.loads((source/'results.json').read_text())['test'][name]
  npz=source/(('test_'+name+'.npz'))
  z=np.load(npz)
  assert np.array_equal(z['ids'],a['ids']) and np.array_equal(z['labels'],a['labels'])
  acc=100*(z['logits'].argmax(1)==z['labels']).mean()
  assert abs(acc-v['accuracy']*100)<1e-4
  rows.append(dict(target_mV=t,skip=k,accuracy_pct=acc,reduction_pct=100*v['hardware']['comparison_saving'],source=str(source/'results.json'),route=name))
colors={20:'#277DA8',40:'#9A629E'};markers={4:'o',5:'s',6:'^'}
offsets={(20,4):(0,12),(20,5):(0,-19),(20,6):(-24,-21),(40,4):(-12,12),(40,5):(0,-20),(40,6):(-12,12)}
f,ax=plt.subplots(figsize=(6.6,4.05));f.subplots_adjust(left=.12,right=.98,bottom=.16,top=.83)
for t in [20,40]:
 g=[r for r in rows if r['target_mV']==t];c=colors[t]
 ax.plot([r['reduction_pct'] for r in g],[r['accuracy_pct'] for r in g],color=c,lw=1.8,zorder=3)
 for r in g:
  x,y=r['reduction_pct'],r['accuracy_pct'];k=r['skip']
  ax.plot(x,y,marker=markers[k],ms=7,color=c,mec='white',mew=.8,zorder=4)
  ax.annotate(f'S{k} · {y:.1f}%',(x,y),xytext=offsets[t,k],textcoords='offset points',ha='center',fontsize=9,color=c)
for name,col,ls in [('FP32','#4E5965',(0,(5,3))),('W4A4','#87939E',(0,(2,2)))]:
 y=bases[name];ax.axhline(y,color=col,ls=ls,lw=1.2,zorder=1)
 ax.text(25.5,y+.12,f'{name}  {y:.1f}%',color=col,fontsize=9,va='bottom',bbox=dict(facecolor='white',edgecolor='none',pad=1))
ax.set(xlim=(25,51),ylim=(74.7,83.1),xlabel='SAR comparison reduction (%)',ylabel='Top-1 accuracy (%)')
ax.set_xticks([25,30,35,40,45,50]);ax.set_yticks([75,76,77,78,79,80,81,82,83]);ax.grid(axis='y',color='#E9EDF0',lw=.65)
handles=[Line2D([],[],color=colors[t],lw=1.8,label=f'{t} mV') for t in [20,40]]
handles += [Line2D([],[],color='#596571',marker=markers[k],lw=0,label=f'Skip {k}',ms=6) for k in [4,5,6]]
f.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.955),ncol=5,columnspacing=1.6,handlelength=1.7,fontsize=9)
out=H/'output';out.mkdir(exist_ok=True)
for ext in ['pdf','svg','png']:f.savefig(out/f'wrn_accuracy_sar.{ext}')
with (out/'wrn_accuracy_sar.csv').open('w') as o:
 w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'wrn_accuracy_sar_notes.md').write_text('''# WRN accuracy–SAR figure

CIFAR-100 same 1000 historical test IDs and labels checked for all six points and two baselines; accuracies recomputed from saved predictions. FP32 denotes float32 evaluation of mixed-precision-trained model. W4A4 is digital quantized baseline, not ADC-native fine-tuned control.

20mV/S4 retains its original successful 20-epoch joint-training run. Four points use the BN-calibrated repair; 40mV/S6 uses the new matched BN-only/20-epoch training run. This is a combined-source operating-point summary, not a uniformly BN-calibrated six-point rerun; source paths are recorded in CSV. Zero gating OFF for SA curves; ADC8, 0–1.2V, three mapped convolution layers. Reduction measures SAR comparisons, not whole-conversion skips or net system energy. Connecting lines only join measured discrete points.

Continuous y axis deliberately retained (74.7–83.1%); no break needed to accommodate 82.1% FP32. Single-seed results; no error bars inferred. Not automatically inserted into manuscript.
''')
print(rows)
