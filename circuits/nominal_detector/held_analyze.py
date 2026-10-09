from pathlib import Path
import csv,re,json,sys
R=Path(__file__).resolve().parent
def collect(name):
 s=(R/name/'spectre.out').read_text()
 return {tag:{int(m[1]):list(map(float,m[2].split(',')))for m in re.finditer(tag+r',(\d+),([^\n]+)',s)}for tag in ['HPRE','HCAP','HADC']}
def extract(name,baseline=None):
 data=collect(name);base=collect(baseline)if baseline else None;events=json.loads((R/(name+'_events.json')).read_text());rows=[]
 for slot,(cm,d)in enumerate(events):
  if not slot:continue
  row=dict(job=name,slot=slot,cm_V=cm,d_mV=d*1000,baseline=baseline or 'same_event_HP RE_initial_reference_not_detector_removed')
  for tag in ['HCAP','HADC']:
   ds=[1000*(data[tag][slot][i]-(base[tag][slot][i]if base else data['HPRE'][slot][i]))for i in [0,1]]
   for label,val in zip(['P','N','CM','DIFF'],ds+[sum(ds)/2,ds[0]-ds[1]]):row[tag+'_'+label+'_mV']=val
  rows.append(row)
 return rows
if __name__=='__main__':
 rows=extract(sys.argv[1],sys.argv[2]if len(sys.argv)>2 else None)
 p=R/(sys.argv[1]+'_held_errors.csv')
 with p.open('w')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 for tag in ['HCAP','HADC']:
  print(tag,{k:max(abs(r[tag+'_'+k+'_mV'])for r in rows)for k in ['P','N','CM','DIFF']})
