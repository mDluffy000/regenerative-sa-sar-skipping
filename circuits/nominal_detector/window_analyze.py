from pathlib import Path
import csv,re,json,math,sys
R=Path(__file__).resolve().parent

def read(name):
 s=(R/name/'spectre.out').read_text();assert re.search(r'spectre completes with 0 errors',s,re.I)
 config=json.loads((R/(name+'_config.json')).read_text());target=config['target_mV'];events=json.loads((R/(name+'_events.json')).read_text())
 collect=lambda tag:{int(m[1]):list(map(float,m[2].split(',')))for m in re.finditer(tag+r',(\d+),([^\n]+)',s)}
 bias=collect('BIASED');core=collect('CORECAP');dead=collect('DEADLINE');reset=collect('RESET');en={}
 code=lambda x:1 if x[0]>.96 and x[1]<.24 else(-1 if x[0]<.24 and x[1]>.96 else 0)
 for m in re.finditer(r'ENERGY,(\d+),(\d+),([^\n]+)',s):
  v=[float(x)*1e15 for x in m[3].split(',')];assert abs(v[0]-(v[1]-v[2]))<.001;en[int(m[1]),int(m[2])]=v
 out=[]
 for m in re.finditer(r'RESULT,([^\n]+)',s):
  a=list(map(float,m[1].split(',')));slot=int(a[0]);
  if not slot:continue
  cm,d=events[slot];mv=d*1000;actual=int(a[3]);expected=1 if abs(mv)>target else(-1 if abs(mv)<target else None)
  b=bias[slot];c=core[slot];rr=reset[slot];stable=max((a[4]if actual==1 else a[5])*1e12+.25,config.get('window_close_ps',700)+10)if actual else math.inf
  ce1=1 if mv>target else(-1 if mv<target else 0);ce2=1 if mv>-target else(-1 if mv<-target else 0)
  r=dict(job=name,slot=slot,target_mV=target,cm_V=cm,d_mV=mv,code=actual,outcome={1:'FAR',-1:'NEAR',0:'UNRESOLVED'}[actual],expected_code=expected if expected is not None else'',classification_pass=int(actual==expected)if expected is not None else'',stable_service_ps=stable,deadline985_pass=int(actual!=0 and code(dead[slot])==actual and stable<=985),core_pos_code=code(c[:2]),core_neg_code=code(c[2:]),core_capture_pass=int(code(c[:2])==ce1 and code(c[2:])==ce2)if ce1 and ce2 else'',reset_pass=int(all(abs(v)<.24 for v in rr[:4])and code(rr[4:])==1),biased_cm_pos_V=(b[0]+b[1])/2,biased_diff_pos_mV=(b[0]-b[1])*1000,biased_cm_neg_V=(b[2]+b[3])/2,biased_diff_neg_mV=(b[2]-b[3])*1000,previous_cm_V=events[slot-1][0],previous_d_mV=events[slot-1][1]*1000)
  for ch,label in enumerate(['core_pos','core_neg','logic','bias','input_p','input_n','clock','window','acq']):
   for tag,v in zip(['net_fJ','draw_fJ','returned_fJ'],en[slot,ch]):r[label+'_'+tag]=v
  r['subtotal_draw_fJ']=sum(r[k+'_draw_fJ']for k in ['core_pos','core_neg','logic','bias','clock','window','acq']);r['all_ports_draw_fJ']=r['subtotal_draw_fJ']+r['input_p_draw_fJ']+r['input_n_draw_fJ']
  r['pass']=int(r['classification_pass']==1 and r['deadline985_pass']and r['core_capture_pass']==1 and r['reset_pass']and r['subtotal_draw_fJ']<=80)
  out.append(r)
 assert len(out)==len(events)-1
 return out

def write(n,rs):
 with(R/n).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
if __name__=='__main__':
 names=json.loads((R/sys.argv[1]).read_text());rs=sum([read(n)for n in names],[]);write(sys.argv[1].replace('jobs_','events_').replace('.json','.csv'),rs)
 for name in names:
  x=[r for r in rs if r['job']==name];print(name,'N',len(x),'pass',sum(r['pass']for r in x),'subtotalmax',max(r['subtotal_draw_fJ']for r in x),'allmax',max(r['all_ports_draw_fJ']for r in x),'stablemax',max(r['stable_service_ps']for r in x))
  for r in x:
   if not r['pass']:print('FAIL',{k:r[k]for k in['cm_V','d_mV','code','expected_code','deadline985_pass','core_pos_code','core_neg_code','core_capture_pass','reset_pass','biased_diff_pos_mV','biased_diff_neg_mV']})
