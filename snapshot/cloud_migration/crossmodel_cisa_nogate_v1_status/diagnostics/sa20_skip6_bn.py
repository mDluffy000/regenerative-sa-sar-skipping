"""User-requested single-point follow-up; fixed prior BN repair protocol."""
from pathlib import Path
import sys,json,copy,time,hashlib,tarfile,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
from torch import nn
import run as t
from models import build,attach,prep,runtime
from hardware import Hardware
from deit_data import read_data
O=R/'diagnostics/sa20_skip6_bn_v1'
def main():
 runtime();torch.manual_seed(t.SEED);t.compile_kernel();O.mkdir(exist_ok=False)
 t.OUT=O;t.EPOCHS=5;t.STATE=dict(model='ResNet20_SA20_skip6_BN_repair',status='running',stages={},test={})
 protocol=dict(target_mV=20,threshold_mV=19,skip=6,ADC_bits=8,zero_gating=False,voltage_range=[0,1.2],epochs=5,train_images=2000,validation_images=500,test_images=1000,BN_calibration='one pass of original hardware train IDs, batch32, no augmentation, reset statistics, cumulative average; no optimizer and no labels',finetune='same AdamW1e-5 to1e-6; batch4; CE+0.5*T²*KL T2; frozen digital W4A4 teacher; BN stats frozen, affine and quantizer scales trainable',selection='validation accuracy then CE; epoch0 eligible; freeze before testing; no extensions',scope='Additional user-requested exploratory point after earlier test results; not predeclared primary matrix. SA physical model and existing failed results unchanged.')
 t.save('PROTOCOL.json',protocol)
 manifest=json.loads((R/'resnet/manifest.json').read_text())['sources']
 for n,h in manifest.items():assert t.sha(R/n)==h,n
 t.save('SOURCE_MANIFEST.json',dict(parent_manifest_verified=True,script_sha256=t.sha(Path(__file__)),parent_checkpoint_sha256=t.sha(R/'input/resnet_W4A4.pt'),held_table_sha256=t.sha(R/'held_20.bin')))
 x,y=read_data('train');split=json.loads((R/'resnet/split.json').read_text());ids=np.array(split['train']);valid=np.array(split['validation']);t.save('split.json',split)
 m=build('resnet');h=Hardware(target=20,skip=6);attach(m,h,'resnet');params={n:p.detach().clone() for n,p in m.named_parameters()}
 before=t.evaluate('resnet',m,x,y,valid,h,'before_validation')
 for op in m.modules():
  if isinstance(op,nn.BatchNorm2d):op.reset_running_stats();op.momentum=None;op.train()
 with torch.no_grad():
  for pos in range(0,len(ids),32):
   ix=ids[pos:pos+32];h.context(ix,0);m(prep(x[ix],'resnet'))
 assert all(torch.equal(v,dict(m.named_parameters())[n]) for n,v in params.items())
 after=t.evaluate('resnet',m,x,y,valid,h,'BN_only_validation')
 torch.save(dict(model=m.state_dict(),validation=after),O/'BN_only.pt')
 t.save('BN_CALIBRATION.json',dict(before_validation=before,after_validation=after,all_trainable_parameters_unchanged=True))
 print('BN_CALIBRATION',before['accuracy'],after['accuracy'],flush=True)
 teacher=build('resnet').eval()
 for p in teacher.parameters():p.requires_grad_(False)
 t.train('resnet','SA20_skip6',m,teacher,h,x,y,ids,valid)
 t.save('SELECTION_FROZEN.json',t.STATE['stages'])
 tx,ty=read_data('test');test=np.array(split['test'])
 for name,path in [('BN_only',O/'BN_only.pt'),('BN_finetuned',O/'SA20_skip6_best.pt')]:
  m.load_state_dict(torch.load(path,weights_only=True)['model']);check=t.exact_check('resnet',m,x,ids);t.save(name+'_exact.json',check)
  t.STATE['test'][name]=t.evaluate('resnet',m,tx,ty,test,h,'test_'+name);t.report()
 a=np.load(R/'continuation_20260913/bn_refresh_finetune/test_BN_finetuned_native_ADC.npz');b=np.load(O/'test_BN_finetuned.npz')
 assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels'])
 ca=a['logits'].argmax(1)==a['labels'];cb=b['logits'].argmax(1)==b['labels']
 t.STATE['paired_vs_existing_matched_control']=dict(control_accuracy=float(ca.mean()),accuracy_change_pp=float((cb.astype(float)-ca).mean()*100),wrong_to_correct=int((~ca&cb).sum()),correct_to_wrong=int((ca&~cb).sum()),same_test_ids=True)
 for n,sha in manifest.items():assert t.sha(R/n)==sha,n
 t.STATE['status']='complete';t.report();t.status('complete')
 hashes={p.name:t.sha(p) for p in O.iterdir() if p.is_file()};t.save('SHA256.json',hashes)
 with tarfile.open(R/'sa20_skip6_bn_v1_delivery.tar.gz','w:gz') as tf:
  for p in O.iterdir():
   if p.is_file():tf.add(p,arcname=p.name)
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():(O/'FAILURE.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2))
  raise
