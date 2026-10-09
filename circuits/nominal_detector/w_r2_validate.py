from w_round2 import direct,R
import json,re
if __name__=='__main__':
 cms=json.loads((R.parent/'wide_cm/cm_grid.json').read_text());names=[]
 for held in [False,True]:
  for t in [20,40]:
   events=[(cm,d/1000)for cm in cms for d in[0,-t-4,-t+4,t-4,t+4]if abs(d)<=2000*min(cm,1.2-cm)+1e-7]
   for i in range(0,len(events),80):
    n=f'w_r2full_{"held" if held else "ideal"}_{t}_{i//80}';direct(n,[(.6,0)]+events[i:i+80],t,held);names.append(n)
 (R/'jobs_wr2full.json').write_text(json.dumps(names))
 names=[]
 for t in[20,40]:
  n=f'w_r2_removed_{t}';ev=json.loads((R/f'w_r2_held_{t}_events.json').read_text());direct(n,ev,t,True)
  p=R/(n+'.scs');s=p.read_text();s='\n'.join(l for l in s.splitlines()if not re.match(r'^(XF |XP |XN |XD |XL |CP |CN |CP2 |CN2 |CQ |CQB )',l))+'\n';p.write_text(s);names.append(n)
 (R/'jobs_wr2baseline.json').write_text(json.dumps(names))
