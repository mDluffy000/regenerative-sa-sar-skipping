"""Validation-only follow-up: resume QAT on all clean training examples."""
import sys,copy,json,time,shutil,traceback,os
from pathlib import Path
P=Path('/root/autodl-tmp/adc_wrn28_10_v1')
sys.path.insert(0,str(P))
import train_wrn as t
import torch,numpy as np
from torch import nn
import torch.nn.functional as F
R=Path(__file__).resolve().parent

def qat(m,teacher,x,y,ids,valid):
 ck=torch.load(P/'W4A4_last.pt',weights_only=True);m.load_state_dict(ck['model'])
 groups=[[],[]]
 for n,p in m.named_parameters():groups[int('scale' in n)].append(p)
 opt=torch.optim.AdamW([dict(params=groups[0],weight_decay=.01),dict(params=groups[1],weight_decay=0.)],lr=5e-6)
 opt.load_state_dict(ck['optimizer'])
 for g in opt.param_groups:g['lr']=5e-6;g['initial_lr']=5e-6
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20,eta_min=1e-6)
 prev=torch.load(P/'W4A4_best.pt',weights_only=True);bestval=prev['validation'];best=(bestval['accuracy'],-bestval['CE']);bestep=prev['epoch'];history=[]
 torch.save(prev,R/'W4A4_best.pt');teacher.eval()
 for p in teacher.parameters():p.requires_grad_(False)
 for ep in range(21,41):
  torch.manual_seed(t.CFG['seed']+200+ep);order=ids[torch.randperm(len(ids)).numpy()];m.train();t.freeze_bn(m);begin=time.monotonic();correct=0;loss_sum=0.
  for pos in range(0,len(ids),32):
   if pos%512==0:t.status('W4A4_extension',epoch=ep,epochs=40,seen=pos,total=len(ids),seconds=time.monotonic()-begin)
   ix=order[pos:pos+32];a=t.prep(x[ix],True);labels=y[ix].cuda();opt.zero_grad(set_to_none=True);z=m(a)
   with torch.no_grad():target=teacher(a)
   loss=F.cross_entropy(z,labels)+2.*F.kl_div(F.log_softmax(z/2,1),F.softmax(target/2,1),reduction='batchmean')
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
   with torch.no_grad():
    for op in m.modules():
     if isinstance(op,t.QuantOp):op.activation_scale.clamp_(min=1e-8);op.weight_scale.clamp_(min=1e-8)
   correct+=int((z.detach().argmax(1)==labels).sum());loss_sum+=float(loss.detach())*len(ix)
  scheduler.step();v=t.evaluate(m,x,y,valid);score=(v['accuracy'],-v['CE'])
  if score>best:best=score;bestval=v;bestep=ep;torch.save(dict(model=m.state_dict(),epoch=ep,validation=v),R/'W4A4_best.pt')
  torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=ep),R/'W4A4_last.pt')
  history.append(dict(epoch=ep,validation=v,train_accuracy=correct/len(ids),loss=loss_sum/len(ids),seconds=time.monotonic()-begin,lr=opt.param_groups[0]['lr']))
  t.save('W4A4_extension_history.json',history);t.RESULTS['stages']['W4A4']=dict(validation=bestval,best_epoch=bestep,epochs_finished=ep);t.report('running')
 m.load_state_dict(torch.load(R/'W4A4_best.pt',weights_only=True)['model']);m.eval();return m

def main():
 t.R=R;t.START=time.monotonic();t.runtime();torch.manual_seed(t.CFG['seed'])
 with (R/'RUN_STARTED.json').open('x') as f:json.dump(dict(utc=t.now(),pid=os.getpid()),f)
 t.CFG.update(qat_images=44976,qat_epochs=40,qat_extension=dict(epochs=20,start='parent epoch20 model and AdamW moments',lr_start=5e-6,lr_end=1e-6,training_expanded_from=10000,training_expanded_to=44976,reason='validation-only follow-up after 5.68pp quantization gap',parent_best_is_candidate=True),maximum_wall_seconds=7200)
 parent=json.loads((P/'results.json').read_text());t.RESULTS.clear();t.RESULTS.update(status='running',config=t.CFG,stages={'floating':parent['stages']['floating']})
 sources=list(P.glob('*.py'))+list(P.glob('*.cu'))+[P/'floating_best.pt',P/'W4A4_best.pt',P/'W4A4_last.pt',P/'calibration.json',P/'split.json',P/'pilot_split.json',R/'extend.py',t.BASE/'model.py',t.BASE/'data.py',t.BASE/'split.json']
 manifest={str(p):t.sha(p) for p in sources};t.save('manifest.json',dict(sources=manifest,torch=torch.__version__,cuda=torch.version.cuda));t.save('protocol.json',t.CFG)
 split=json.loads((P/'split.json').read_text());pilot=json.loads((P/'pilot_split.json').read_text());t.save('split.json',split)
 x,y=t.read_data('train');ids=np.array(split['train']);valid=np.array(split['validation']);assert len(ids)==44976 and len(valid)==5000 and not set(ids)&set(valid)
 qids=ids;hids=np.array(pilot['hardware_train']);hval=np.array(pilot['hardware_validation']);pilot['qat']=ids.tolist();t.save('pilot_split.json',pilot)
 fp=t.WRN().cuda();fp.load_state_dict(torch.load(P/'floating_best.pt',weights_only=True)['model']);fp.eval()
 scales=json.loads((P/'calibration.json').read_text())['scales'];q=t.quantize(copy.deepcopy(fp),scales);q=qat(q,fp,x,y,ids,valid)
 gap=100*(t.RESULTS['stages']['floating']['validation']['accuracy']-t.RESULTS['stages']['W4A4']['validation']['accuracy'])
 if gap>5:t.gate(f'Extended QAT gap {gap:.2f} pp exceeds unchanged 5 pp gate.');return
 t.RESULTS['initial_exact_check']=t.exact_check(q,x,qids);teacher=copy.deepcopy(q).eval();models={}
 for name,mode in [('adc_control','adc'),('sa_skip5','sa')]:
  hw=t.Hardware(mode);m=copy.deepcopy(q);t.attach(m,hw);models[name]=t.train(name,m,x,y,hids,hval,3,teacher,hw)
 t.save('SELECTION_FROZEN.json',dict(utc=t.now(),stages=t.RESULTS['stages']));t.RESULTS['final_exact_check']=t.exact_check(models['sa_skip5'],x,qids)
 tx,ty=t.read_data('test');test=np.array(json.loads((t.BASE/'test_ids.json').read_text()));t.save('test_ids.json',test.tolist());t.RESULTS['test']={}
 for name,m,mode in [('floating',fp,None),('W4A4',q,None),('native_ADC',q,'adc'),('disturbance_only',q,'disturbance'),('direct_skip5',q,'sa'),('ordinary_ADC',models['adc_control'],'adc'),('ordinary_skip5',models['adc_control'],'sa'),('aware_skip5',models['sa_skip5'],'sa')]:
  hw=t.Hardware(mode) if mode else None
  if hw:t.attach(m,hw)
  t.RESULTS['test'][name]=t.evaluate(m,tx,ty,test,hw,'test_'+name,t.CFG['test_perturbation_seed']);t.report('running')
 b=np.load(R/'test_aware_skip5.npz');cb=b['logits'].argmax(1)==b['labels'];t.RESULTS['paired']={}
 for name in ['direct_skip5','ordinary_skip5']:
  a=np.load(R/('test_'+name+'.npz'));assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels']);ca=a['logits'].argmax(1)==a['labels']
  t.RESULTS['paired'][name]=dict(gain_pp=float((cb.astype(float)-ca).mean()*100),wrong_to_correct=int((~ca&cb).sum()),correct_to_wrong=int((ca&~cb).sum()))
 for path,h in manifest.items():assert t.sha(Path(path))==h,path
 t.RESULTS['elapsed_seconds']=time.monotonic()-t.START;t.report('complete');t.save('status.json',dict(state='complete',utc=t.now()))
 (R/'SHA256SUMS').write_text(''.join(t.sha(p)+'  '+p.name+'\n' for p in sorted(R.iterdir()) if p.is_file() and p.name not in ['SHA256SUMS','pipeline.log']))
if __name__=='__main__':
 if (R/'RUN_STARTED.json').exists():raise SystemExit('Existing extension preserved')
 try:main()
 except Exception:
  t.RESULTS['reason']=traceback.format_exc();t.report('failed');t.save('status.json',dict(state='failed',reason=t.RESULTS['reason'],utc=t.now()));raise
