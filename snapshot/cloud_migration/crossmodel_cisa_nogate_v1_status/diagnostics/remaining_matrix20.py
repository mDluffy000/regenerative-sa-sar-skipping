from pathlib import Path
import sys,json,shutil,traceback,tarfile
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
import run as t
from models import build,runtime
from hardware import Hardware
from deit_data import read_data
O=R/'diagnostics/remaining_matrix20_v1'
exec(compile((Path(__file__).parent/'skip5_extend_loop.py').read_text(),'skip5_extend_loop.py','exec'),t.__dict__)
routes=[('SA20_skip4',20,4,R/'continuation_20260913/bn_refresh_finetune'),('SA20_skip6',20,6,R/'diagnostics/sa20_skip6_bn_v1'),('SA40_skip6',40,6,R/'continuation_20260913/bn_refresh_finetune'),('SA40_skip4',40,4,None)]
def main():
 runtime();torch.manual_seed(t.SEED);t.compile_kernel();O.mkdir(exist_ok=False)
 t.OUT=O;t.EPOCHS=20;t.STATE=dict(model='ResNet20_remaining_matrix20',status='running',stages={},test={})
 manifest=json.loads((R/'resnet/manifest.json').read_text())['sources']
 for n,h in manifest.items():assert t.sha(R/n)==h,n
 t.save('PROTOCOL.json',dict(scope='User-requested exploratory extension after viewing 5-epoch tests; not blind preregistered evaluation.',total_epochs=20,additional_epochs={"resumed_points":15,"new_40_skip4":20},learning_rate='Preserve first 5 epochs and AdamW optimizer; continue at terminal learning rate 1e-6 for epochs 6-20 in the three resumed arms; not a fresh 20-epoch cosine schedule.',BN='Retain route-specific calibrated BN statistics, frozen during fine-tuning.',selection='Best validation accuracy then CE across epoch0-20; freeze all four branches before testing.',train=2000,validation=500,test=1000,zero_gating=False,targets_mV=[20,40],routes=[(n,a,k) for n,a,k,_ in routes],new_point_schedule='40mV skip4: train-only BN calibration then fresh20 cosine1e-5 to1e-6; existing points: preserve5 then15 at1e-6',control='Reuse existing native_ADC20 and two skip5 results; no rerun; disclose schedule differences',source_hashes=manifest))
 split=json.loads((R/'resnet/split.json').read_text());t.save('split.json',split)
 x,y=read_data('train');ids=np.array(split['train']);valid=np.array(split['validation'])
 teacher=build('resnet').eval()
 for p in teacher.parameters():p.requires_grad_(False)
 for name,target,k,parent in routes:
  if parent is not None:
   for suffix in ['_last.pt','_best.pt','_history.json']:shutil.copy2(parent/(name+suffix),O/(name+suffix))
  h=Hardware(target=target,skip=k)
  m=build('resnet')
  if parent is None:
   from models import attach,prep
   attach(m,h,'resnet');params={n:p.detach().clone() for n,p in m.named_parameters()}
   before=t.evaluate('resnet',m,x,y,valid,h,name+'_before_BN')
   for op in m.modules():
    if isinstance(op,torch.nn.BatchNorm2d):op.reset_running_stats();op.momentum=None;op.train()
   with torch.no_grad():
    for pos in range(0,len(ids),32):
     ix=ids[pos:pos+32];h.context(ix,0);m(prep(x[ix],'resnet'))
   assert all(torch.equal(v,dict(m.named_parameters())[n]) for n,v in params.items())
   after=t.evaluate('resnet',m,x,y,valid,h,name+'_after_BN')
   torch.save(dict(model=m.state_dict(),validation=after),O/(name+'_BN.pt'))
   t.save(name+'_BN_CALIBRATION.json',dict(before=before,after=after,trainable_parameters_unchanged=True))
  t.train_resume('resnet',name,m,teacher,h,x,y,ids,valid)
  m.load_state_dict(torch.load(O/(name+'_best.pt'),weights_only=True)['model'])
  t.save(name+'_exact.json',t.exact_check('resnet',m,x,ids));del m,h;torch.cuda.empty_cache()
 t.save('SELECTION_FROZEN.json',t.STATE['stages'])
 tx,ty=read_data('test');test=np.array(split['test'])
 for name,target,k,parent in routes:
  m=build('resnet');m.load_state_dict(torch.load(O/(name+'_best.pt'),weights_only=True)['model'])
  h=Hardware(target=target,skip=k)
  t.STATE['test'][name]=t.evaluate('resnet',m,tx,ty,test,h,'test_'+name);t.report()
  del m,h;torch.cuda.empty_cache()
 for n,h in manifest.items():assert t.sha(R/n)==h,n
 t.STATE['status']='complete';t.report();t.status('complete')
 t.save('SHA256.json',{p.name:t.sha(p) for p in O.iterdir() if p.is_file()})
 with tarfile.open(R/'remaining_matrix20_v1_delivery.tar.gz','w:gz') as tf:
  for p in O.iterdir():
   if p.is_file():tf.add(p,arcname=p.name)
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():(O/'FAILURE.json').write_text(traceback.format_exc())
  raise
