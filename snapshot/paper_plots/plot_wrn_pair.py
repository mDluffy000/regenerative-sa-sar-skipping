from pathlib import Path
import runpy,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent
D=runpy.run_path(str(H/'plot_wrn.py'));plt.close('all')
rows=D['rows'];base=D['bases'];colors=D['colors'];markers=D['markers'];old=D['old'];B=D['B']
# Matched BN-calibrated parent versus its error-aware-trained descendant.
new=D['new'];plain=[];aware=[]
for k in (4,5,6):
 for phase,values in [('BN',plain),('best',aware)]:
  name=f'combined_40_skip{k}_{phase}'
  z=np.load(B/('adc_wrn40_skip6_paired_v1' if k==6 else 'adc_wrn_bn_repair_v1')/f'test_{name}.npz')
  assert np.array_equal(z['ids'],D['a']['ids']) and np.array_equal(z['labels'],D['a']['labels'])
  values.append(float(100*(z['logits'].argmax(1)==z['labels']).mean()))
f=plt.figure(figsize=(7.2,3.35));gs=f.add_gridspec(1,2,width_ratios=[1.16,1],left=.075,right=.985,bottom=.22,top=.84,wspace=.30)
ax=f.add_subplot(gs[0]);top=f.add_subplot(gs[1])
off={(20,4):(0,7),(20,5):(-6,-13),(20,6):(-10,-14),(40,4):(-6,7),(40,5):(0,-13),(40,6):(-8,8)}
for t in (20,40):
 g=[r for r in rows if r['target_mV']==t]
 ax.plot([r['reduction_pct'] for r in g],[r['accuracy_pct'] for r in g],c=colors[t],lw=1.5)
 for r in g:
  k=r['skip'];x=r['reduction_pct'];y=r['accuracy_pct']
  ax.plot(x,y,marker=markers[k],c=colors[t],ms=5,mec='white',mew=.5)
  ax.annotate(f'S{k}',(x,y),xytext=off[t,k],textcoords='offset points',ha='center',fontsize=7.5,color=colors[t])
for name,c,ls in [('FP32','#52616E','--'),('W4A4','#84929D',':')]:
 ax.axhline(base[name],c=c,ls=ls,lw=1)
 ax.text(25.5,base[name]+.14,f'{name}: {base[name]:.1f}%',fontsize=7,c=c)
ax.set(xlim=(25,51),ylim=(74.5,83.2),xlabel='SAR comparison reduction (%)',ylabel='Top-1 accuracy (%)')
ax.set_xticks([25,30,35,40,45,50]);ax.set_yticks([75,77,79,81,83]);ax.grid(axis='y',color='#E8ECEF',lw=.5)
ax.legend(handles=[Line2D([],[],c=colors[t],label=f'{t} mV') for t in (20,40)],loc='lower left',bbox_to_anchor=(-.03,1.04),ncol=2,fontsize=8)
top.plot([4,5,6],aware,color=colors[40],marker='o',ms=5,lw=1.5,label='SA disturbance & encoding error training')
top.plot([4,5,6],plain,color='#6E7C89',marker='s',ms=4.5,lw=1.3,ls='--',label='No error training')
top.set(xlim=(3.7,6.3),ylim=(66.5,79),xlabel='Skipped SAR bits, k',ylabel='Top-1 accuracy (%)')
top.set_xticks([4,5,6]);top.set_yticks([68,70,72,74,76,78]);top.grid(axis='y',color='#E8ECEF',lw=.5)
for k,y in zip([4,5,6],aware):top.annotate(f'{y:.1f}%',(k,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8,color=colors[40])
for k,y in zip([4,5,6],plain):top.annotate(f'{y:.1f}%',(k,y),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=8,color='#52616E')
top.legend(loc='lower left',bbox_to_anchor=(-.02,1.03),fontsize=6.8,handlelength=1.8)
f.text(.295,.035,'(a) WRN operating points',ha='center',fontsize=9)
f.text(.79,.035,'(b) Training benefit at 40 mV',ha='center',fontsize=9)
# Restrained depth on data lines and markers; baselines and axes remain flat.
for axis in (ax,top):
 for line in axis.lines:
  if line.get_color() in [colors[20],colors[40],'#6E7C89']:
   line.set_path_effects([pe.SimpleLineShadow(offset=(.65,-.8),shadow_color='#3A4652',alpha=.18,rho=1.05),pe.Normal()])
   if line.get_marker() not in ['None',None,'',' ']:
    line.set_markeredgecolor('white');line.set_markeredgewidth(.65)
out=H/'output'
for ext in ['pdf','svg','png']:f.savefig(out/f'wrn_two_panel.{ext}')
(out/'wrn_two_panel_caption.tex').write_text(r"""\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/wrn_two_panel.pdf}
  \caption{WRN-28-10 results on the same 1,000-image CIFAR-100 test subset. (a) Accuracy versus SAR comparison reduction with FP32 and digital W4A4 references. (b) At 40 mV, train-set BN calibration alone versus the same calibrated models after 20 epochs of error-aware fine-tuning. Both arms include modeled SA voltage perturbations and prefix errors during inference. All three points in (b) compare each BN-calibrated checkpoint to its trained descendant. Panel (a) retains the prior 20 mV/skip-4 run and uses BN-repaired or matched-run results for the other five points. ADC8, zero gating disabled, and three mapped convolution layers are used.}
  \label{fig:wrn-results}
\end{figure*}
""")
(out/'wrn_two_panel_comparison.json').write_text(json.dumps(dict(skip=[4,5,6],BN_only_accuracy=plain,trained_accuracy=aware,scope='BN-calibrated starting checkpoints versus their trained descendants; complete physical behavior in both arms'),indent=2))
