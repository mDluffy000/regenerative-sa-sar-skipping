from pathlib import Path
import json,tarfile,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent;R=H.parent;B=R/'cloud_migration/remote_results'
plt.style.use(H/'styles/iscas.mplstyle')
res=json.loads((B/'remaining_matrix20_v1/results.json').read_text())
with tarfile.open(R/'cloud_migration/skip5_extend20_v1_delivery.tar.gz') as t:old=json.load(t.extractfile('results.json'))
dei=json.loads((B/'adc_crossmodel_complete_20260913/deit/results.json').read_text())
rows=[]
for model in ['ResNet20','DeiT']:
 for target in [20,40]:
  for k in [4,5,6]:
   if model=='ResNet20':v=(old if k==5 else res)['test'][f'SA{target}_skip{k}']
   else:v=dei['test'][f'aware_{target}_skip{k}']
   rows.append(dict(model=model,target_mV=target,skip=k,accuracy_pct=v['accuracy']*100,reduction_pct=v['hardware']['comparison_saving']*100))
colors={'ResNet20':'#277DA8','DeiT':'#9A629E'};marks={4:'o',5:'s',6:'^'}
f,(ax,ar)=plt.subplots(1,2,figsize=(9.0,3.8));f.subplots_adjust(left=.075,right=.98,bottom=.23,top=.78,wspace=.25)
offsets={('ResNet20',20,4):(0,9),('ResNet20',20,5):(-4,9),('ResNet20',20,6):(14,1),('ResNet20',40,4):(-19,-10),('ResNet20',40,5):(10,8),('ResNet20',40,6):(0,9),('DeiT',20,4):(0,9),('DeiT',20,5):(0,9),('DeiT',20,6):(0,9),('DeiT',40,4):(0,-15),('DeiT',40,5):(0,-15),('DeiT',40,6):(-8,-15)}
handles=[]
for model in ['ResNet20','DeiT']:
 for target in [20,40]:
  g=[r for r in rows if r['model']==model and r['target_mV']==target];c=colors[model];ls='-' if target==20 else '--'
  line,=ax.plot([r['reduction_pct'] for r in g],[r['accuracy_pct'] for r in g],c=c,ls=ls,lw=1.6)
  line.set_path_effects([pe.SimpleLineShadow(offset=(.5,-.7),alpha=.14),pe.Normal()])
  for r in g:
   k=r['skip'];x=r['reduction_pct'];y=r['accuracy_pct']
   dot,=ax.plot(x,y,marker=marks[k],ms=6.3,mfc=c if target==20 else 'white',mec=c,mew=1,lw=0)
   dot.set_path_effects([pe.SimpleLineShadow(offset=(.5,-.7),alpha=.14),pe.Normal()])
   ax.annotate(f'S{k}',(x,y),xytext=offsets[model,target,k],textcoords='offset points',ha='center',fontsize=7,color=c)
  handles.append(Line2D([],[],color=c,ls=ls,label=f'{model}, {target} mV',lw=1.6))
ax.set(xlim=(14,50),ylim=(50,76),xlabel='SAR comparison reduction (%)',ylabel='Top-1 accuracy (%)')
ax.set_xticks([15,20,25,30,35,40,45,50]);ax.set_yticks([50,55,60,65,70,75]);ax.grid(axis='y',color='#E8ECEF',lw=.6)
ax.text(.97,.96,'Digital references (same test subset)\nResNet20: FP32 69.9%  |  W4A4 67.0%\nDeiT: FP32 79.7%  |  W4A4 75.5%',transform=ax.transAxes,ha='right',va='top',fontsize=6.4,color='#52616E',linespacing=1.55,bbox=dict(boxstyle='round,pad=.5',facecolor='white',edgecolor='#DFE5EA',alpha=.96))
ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.02,1.02),ncol=2,columnspacing=1.1,fontsize=7)
# Additional panel: same 20mV hardware behavior, no adaptation vs trained.
bn4=json.loads((B/'adc_crossmodel_complete_20260913/bn_refresh_finetune/results.json').read_text())['test']['BN_only_SA20_skip4']
bn5=json.loads((B/'sa20_skip5_bn_v1/results.json').read_text())['test']['BN_only']
bn6=json.loads((B/'sa20_skip6_bn_v1/results.json').read_text())['test']['BN_only']
controls={'ResNet20':[100*v['accuracy'] for v in [bn4,bn5,bn6]],'DeiT':[100*dei['test'][f'direct_20_skip{k}']['accuracy'] for k in [4,5,6]]}
comp=[];leg=[]
for model in ['ResNet20','DeiT']:
 c=colors[model]
 trained=[r['accuracy_pct'] for r in rows if r['model']==model and r['target_mV']==20]
 for trained_flag,ys in [(True,trained),(False,controls[model])]:
  ls='-' if trained_flag else '--';marker='o' if trained_flag else 's'
  line,=ar.plot([4,5,6],ys,color=c,ls=ls,marker=marker,ms=5,mfc=c if trained_flag else 'white',mec=c,lw=1.5)
  line.set_path_effects([pe.SimpleLineShadow(offset=(.5,-.7),alpha=.14),pe.Normal()])
  leg.append(Line2D([],[],color=c,ls=ls,marker=marker,ms=4,label=model+(' / trained' if trained_flag else ' / no error training')))
  for k,y in zip([4,5,6],ys):
   dy=8 if trained_flag or (model == "DeiT" and k == 6) else -14
   ar.annotate(f'{y:.1f}%',(k,y),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=7,color=c)
   comp.append(dict(model=model,skip=k,error_training=trained_flag,accuracy_pct=y))
ar.set(xlim=(3.75,6.25),ylim=(20,79),xlabel='Skipped SAR bits, k',ylabel='Top-1 accuracy (%)')
ar.set_xticks([4,5,6]);ar.set_yticks([20,30,40,50,60,70]);ar.grid(axis='y',color='#E8ECEF',lw=.6)
ar.legend(handles=leg,loc='lower left',bbox_to_anchor=(-.04,1.02),ncol=2,columnspacing=.8,fontsize=6.6,handlelength=1.7)
f.text(.285,.035,'(a) Accuracy–SAR trade-off',ha='center',fontsize=10)
f.text(.775,.035,'(b) Error-training benefit at 20 mV',ha='center',fontsize=10)
out=H/'output'
for ext in ['png','pdf','svg']:f.savefig(out/f'resnet_deit_two_panel.{ext}')
with (out/'resnet_deit_tradeoff.csv').open('w') as o:
 w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'resnet_deit_tradeoff_notes.md').write_text('''# ResNet20 and DeiT, four curves
All points use SA disturbance and encoding error-aware fine-tuning, W4A4, ADC8, 0–1.2 V mapping, zero gating OFF, three mapped layers, historical 1000-image test subsets. All have completed 20 epochs; ResNet20 five points use original5+15 continuation at terminal learning rate, while new40mV/skip4 and DeiT use fresh20 schedules. Do not claim identical schedules or architecture-wide generality. S4/S5/S6 denote skipped SAR bits, not ADC bit precision. Connecting lines join discrete evaluated points; savings are SAR comparisons, not whole ADC conversions or net system energy. Source files: remaining_matrix20_v1/results.json, skip5_extend20_v1_delivery.tar.gz:results.json, adc_crossmodel_complete_20260913/deit/results.json. Figure not inserted into manuscript automatically.
''')

(out/'resnet_deit_20mV_comparison.json').write_text(json.dumps(comp,indent=2))
