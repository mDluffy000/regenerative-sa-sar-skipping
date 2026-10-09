from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parents[1];D=R/'results/final';O=R/'generated';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':6.5,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':5.7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42})
f,axs=plt.subplots(2,2,figsize=(3.5,4.9));f.subplots_adjust(left=.12,right=.985,bottom=.12,top=.89,wspace=.40,hspace=1.08)
a,b,c,d=axs.flat;blue='#277DA8';purple='#9A629E';gray='#6E7C89';marks={4:'o',5:'s',6:'^'}
for ax in axs.flat:ax.grid(axis='y',color='#E8ECEF',lw=.5);ax.set_ylabel('Top-1 accuracy (%)',labelpad=2);ax.tick_params(length=2,pad=2)
def rows(name):return list(csv.DictReader((D/name).open()))
w=rows('wrn_accuracy_sar.csv')
for t,col in [(20,blue),(40,purple)]:
 g=[r for r in w if int(r['target_mV'])==t];a.plot([float(r['reduction_pct']) for r in g],[float(r['accuracy_pct']) for r in g],color=col,lw=1)
 for r in g:
  k=int(r['skip']);x=float(r['reduction_pct']);y=float(r['accuracy_pct']);a.plot(x,y,marker=marks[k],ms=4,color=col,mec='white',mew=.3);a.annotate(f'S{k}',(x,y),xytext=(0,5 if (t,k)!=(20,5) else -9),textcoords='offset points',ha='center',fontsize=5.8,color=col)
for y,name,ls in [(82.1,'FP32','--'),(78.4,'W4A4',':')]:a.axhline(y,color=gray,lw=.7,ls=ls);a.text(25.5,y+.22,f'{name}: {y}%',fontsize=5.5,color=gray)
a.set(xlim=(24,51),ylim=(74.5,83.5),xticks=[25,35,45],yticks=[75,77,79,81,83],xlabel='SAR comparison\nreduction (%)');a.legend(handles=[Line2D([],[],color=co,label=f'{t} mV') for t,co in [(20,blue),(40,purple)]],loc='lower left',bbox_to_anchor=(-.03,1.02),frameon=False,ncol=2,columnspacing=.7,handlelength=1.2)
v=json.loads((D/'wrn_two_panel_comparison.json').read_text())
for key,col,mark,ls,label in [('trained_accuracy',purple,'o','-','SA disturbance &\nencoding error training'),('BN_only_accuracy',gray,'s','--','No error training')]:
 ys=v[key];b.plot([4,5,6],ys,color=col,marker=mark,ms=3.5,lw=1,ls=ls,label=label)
 for k,y in zip([4,5,6],ys):b.annotate(f'{y:.1f}%',(k,y),xytext=(0,5 if key=='trained_accuracy' else -10),textcoords='offset points',ha='center',fontsize=5.8,color=col)
b.set(xlim=(3.7,6.3),ylim=(66,79),xticks=[4,5,6],yticks=[68,72,76,78],xlabel='Skipped SAR bits, $k$');b.legend(loc='lower left',bbox_to_anchor=(-.05,1.01),frameon=False,handlelength=1.2)
g=rows('resnet_deit_tradeoff.csv');handles=[]
for model,col in [('ResNet20',blue),('DeiT',purple)]:
 for t in [20,40]:
  rr=[r for r in g if r['model']==model and int(r['target_mV'])==t];ls='-' if t==20 else '--';c.plot([float(r['reduction_pct']) for r in rr],[float(r['accuracy_pct']) for r in rr],color=col,lw=1,ls=ls)
  for r in rr:
   k=int(r['skip']);x=float(r['reduction_pct']);y=float(r['accuracy_pct']);c.plot(x,y,marker=marks[k],ms=4,mec=col,mfc=col if t==20 else 'white',mew=.6)
   dx,dy=(0,5) if t==20 else (0,-10)
   if model=='ResNet20' and t==20:dx,dy={4:(-5,6),5:(0,-10),6:(6,5)}[k]
   if model=='DeiT' and t==40 and k==6:dx,dy=(4,-17)
   if model=='ResNet20' and t==20 and k==5:dx,dy=(-8,-17)
   if model=='ResNet20' and t==40 and k==4:dx,dy=(-8,10)
   if model=='ResNet20' and t==40 and k==5:dx,dy=(0,6)
   c.annotate(f'S{k}',(x,y),xytext=(dx,dy),textcoords='offset points',ha='center',fontsize=5.4,color=col)
  handles.append(Line2D([],[],color=col,ls=ls,label=f'{model}, {t} mV'))
c.set(xlim=(13,51),ylim=(49,77),xticks=[15,25,35,45],yticks=[50,55,60,65,70,75],xlabel='SAR comparison\nreduction (%)');c.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.06,1.20),frameon=False,handlelength=2.2,ncol=1,labelspacing=.25)
c.text(0,1.035,'FP32 / W4A4 (%)\nResNet20: 69.9 / 67.0\nDeiT: 79.7 / 75.5',transform=c.transAxes,fontsize=5.3,color=gray,va='bottom')
comp=json.loads((D/'resnet_deit_20mV_comparison.json').read_text())
for model,col in [('ResNet20',blue),('DeiT',purple)]:
 for trained in [True,False]:
  rr=[r for r in comp if r['model']==model and r['error_training']==trained];ys=[r['accuracy_pct'] for r in rr];d.plot([4,5,6],ys,color=col,ls='-' if trained else '--',marker='o' if trained else 's',ms=3.5,mfc=col if trained else 'white',lw=1,label=model+(' / trained' if trained else ' / no training'))
  for k,y in zip([4,5,6],ys):d.annotate(f'{y:.1f}%',(k,y),xytext=(0,5 if trained or (model=='DeiT' and k==6) else -10),textcoords='offset points',ha='center',fontsize=5.6,color=col)
d.set(xlim=(3.7,6.3),ylim=(20,79),xticks=[4,5,6],yticks=[20,30,40,50,60,70],xlabel='Skipped SAR bits, $k$');d.legend(loc='lower left',bbox_to_anchor=(-.08,1.20),frameon=False,handlelength=1.2,labelspacing=.25)
for ax,title in zip(axs.flat,['(a) WRN trade-off','(b) WRN training, 40 mV','(c) ResNet20 / DeiT trade-off','(d) Training benefit, 20 mV']):ax.text(.5,-.32,title,ha='center',va='top',transform=ax.transAxes,fontsize=6.3)
f.savefig(O/'results.pdf');f.savefig(O/'results.png',dpi=350)
