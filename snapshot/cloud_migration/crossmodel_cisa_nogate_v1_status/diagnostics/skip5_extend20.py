from pathlib import Path
import sys,json,shutil,traceback,tarfile
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
import run as t
from models import build,runtime
from hardware import Hardware
from deit_data import read_data
O=R/'diagnostics/skip5_extend20_v1'
exec(compile((Path(__file__).parent/'skip5_extend_loop.py').read_text(),'skip5_extend_loop.py','exec'),t.__dict__)
routes=[('native_ADC',None,R/'continuation_20260913/bn_refresh_finetune'),('SA20_skip5',20,R/'diagnostics/sa20_skip5_bn_v1'),('SA40_skip5',40,R/'diagnostics/sa40_skip5_bn_v1')]
def main():
 runtime();torch.manual_seed(t.SEED);t.compile_kernel();O.mkdir(exist_ok=False)
 t.OUT=O;t.EPOCHS=20;t.STATE=dict(model='ResNet20_skip5_BN_extend20',status='running',stages={},test={})
 manifest=json.loads((R/'resnet/manifest.json').read_text())['sources']
 for n,h in manifest.items():assert t.sha(R/n)==h,n
 t.save('PROTOCOL.json',dict(scope='User-requested exploratory extension after viewing 5-epoch tests; not blind preregistered evaluation.',total_epochs=20,additional_epochs=15,learning_rate='Preserve first 5 epochs and AdamW optimizer; continue at terminal learning rate 1e-6 for epochs 6-20 in all three arms; not a fresh 20-epoch cosine schedule.',BN='Retain route-specific calibrated BN statistics, frozen during fine-tuning.',selection='Best validation accuracy then CE across epoch0-20; freeze all three branches before testing.',train=2000,validation=500,test=1000,zero_gating=False,targets_mV=[20,40],skip=5,source_hashes=manifest))
 split=json.loads((R/'resnet/split.json').read_text());t.save('split.json',split)
 x,y=read_data('train');ids=np.array(split['train']);valid=np.array(split['validation'])
 teacher=build('resnet').eval()
 for p in teacher.parameters():p.requires_grad_(False)
 for name,target,parent in routes:
  for suffix in ['_last.pt','_best.pt','_history.json']:shutil.copy2(parent/(name+suffix),O/(name+suffix))
  h=Hardware('adc',disturbance=False) if target is None else Hardware(target=target,skip=5)
  m=build('resnet');t.train_resume('resnet',name,m,teacher,h,x,y,ids,valid)
  m.load_state_dict(torch.load(O/(name+'_best.pt'),weights_only=True)['model'])
  t.save(name+'_exact.json',t.exact_check('resnet',m,x,ids));del m,h;torch.cuda.empty_cache()
 t.save('SELECTION_FROZEN.json',t.STATE['stages'])
 tx,ty=read_data('test');test=np.array(split['test'])
 for name,target,parent in routes:
  m=build('resnet');m.load_state_dict(torch.load(O/(name+'_best.pt'),weights_only=True)['model'])
  h=Hardware('adc',disturbance=False) if target is None else Hardware(target=target,skip=5)
  t.STATE['test'][name]=t.evaluate('resnet',m,tx,ty,test,h,'test_'+name);t.report()
  del m,h;torch.cuda.empty_cache()
 for n,h in manifest.items():assert t.sha(R/n)==h,n
 t.STATE['status']='complete';t.report();t.status('complete')
 t.save('SHA256.json',{p.name:t.sha(p) for p in O.iterdir() if p.is_file()})
 with tarfile.open(R/'skip5_extend20_v1_delivery.tar.gz','w:gz') as tf:
  for p in O.iterdir():
   if p.is_file():tf.add(p,arcname=p.name)
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():(O/'FAILURE.json').write_text(traceback.format_exc())
  raise
