"""Train-only WRN baseline, then W4A4, then matched ADC/SA-aware pilot."""
from pathlib import Path
import copy,datetime,hashlib,json,os,time,traceback
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from wrn import WRN,QuantOp,quantize,attach,runtime,prep,BASE,TARGETS
from gpu_hardware import Hardware
from data import read_data,create_split,subset
R=Path(__file__).resolve().parent
CFG=dict(seed=20260911,model='WRN-28-10 no dropout',classes=100,input_resolution=32,
 floating_epochs=200,floating_batch=128,floating_lr=.1,floating_lr_min=.001,
 floating_training_precision='BF16 autocast forward/backward, FP32 stored parameters and FP32 evaluation',
 qat_epochs=20,qat_images=10000,qat_batch=32,qat_lr=5e-5,
 hardware_epochs=3,hardware_images=2000,hardware_batch=4,hardware_lr=1e-5,
 hardware_validation_images=500,validation_images=5000,pilot_test_images=1000,
 minimum_floating_validation_accuracy=.74,maximum_QAT_gap_pp=5.,
 quantization='all29 Conv/Linear W[-7,7] A[-7,7], STE learned scales, FP32 accumulation',
 frozen_batchnorm='running statistics frozen from floating checkpoint throughout QAT and hardware finetuning; affine parameters trainable',
 hardware_targets=TARGETS,ADC_bits=8,raw_FS=192,skip_bits=5,SA_window_LSB=8.,held_sigma_LSB=.03,
 hardware_assumption='same independent synthetic held-error model as DeiT, not physical calibration',
 mapping='128 rows x128 columns, 2-bit cells, four signed activation bits, W+7 offset digits with exact digital correction',
 ordering='independent reads parallel on CUDA; serial returned-code and held-value feedback within each read; reset across array/bit/token/reduction tile',
 partial_analog_coverage=True,validation_perturbation_seed=20261011,test_perturbation_seed=20261010,
 selection='best validation accuracy then lowest CE; include epoch0; tests only after selection frozen',
 maximum_wall_seconds=14400,pretrained_checkpoint=None,
 dataset_history='Same CIFAR-100 dataset has been evaluated previously in project; not a fresh blind benchmark. No external full-CIFAR-trained weights.')
START=None;RESULTS={'status':'running','config':CFG,'stages':{}}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):
 if START and time.monotonic()-START>CFG['maximum_wall_seconds']:raise TimeoutError('Four-hour experiment limit; completed checkpoints preserved')
 save('status.json',dict(state='running',stage=stage,utc=now(),**kw));print(stage,json.dumps(kw),flush=True)
def report(state):
 RESULTS['status']=state;save('results.json',RESULTS)
 s='# WideResNet-28-10 / CIFAR-100\n\n状态：'+state+'。\n\n'
 s+='从随机初始化训练无dropout的WRN-28-10；沿用去重后的官方train训练/验证划分。原生32×32输入。全部29个Conv/Linear采用W4A4，BN、ReLU、残差相加和池化保持数字计算。ADC/SA仅映射三个stage的最后一个block的conv2。\n\n'
 s+='SA窗口8 LSB、独立高斯保持误差σ=0.03 LSB、8-bit ADC、skip5，与DeiT操作点一致。独立read并行；同一read的返回码反馈和服务顺序保持不变。该模型为假设行为模型，不是实测SA拟合。\n\n'
 s+='| 阶段 | 最佳验证精度 | 选中epoch | 已完成epoch |\n|---|---:|---:|---:|\n'
 for n,v in RESULTS['stages'].items():s+=f"| {n} | {100*v['validation']['accuracy']:.2f}% | {v['best_epoch']} | {v.get('epochs_finished',0)} |\n"
 if RESULTS.get('test'):
  s+='\n冻结模型后在相同1,000张测试图上评估：\n\n| 路线 | 准确率 | SAR比较次数 |\n|---|---:|---:|\n'
  for n,v in RESULTS['test'].items():s+=f"| {n} | {100*v['accuracy']:.2f}% | {v.get('hardware',{}).get('comparisons','—')} |\n"
 if RESULTS.get('paired'):s+='\n配对结果：\n```json\n'+json.dumps(RESULTS['paired'],indent=2)+'\n```\n'
 if RESULTS.get('reason'):s+='\n停止原因：'+RESULTS['reason']+'\n'
 s+='\n浮点训练使用BF16自动混合精度，保存参数和评估为FP32；W4A4与CIM整数重构禁用autocast和TF32。量化及硬件训练冻结BN运行统计。比较次数不能直接当全系统净能耗或吞吐收益。公开80%左右精度仅是文献参考，未作为本次结果。\n'
 (R/'REPORT.md').write_text(s)

@torch.no_grad()
def evaluate(m,x,y,ids,hw=None,name=None,seed=None):
 m.eval();zs=[];begin=time.monotonic();batch=4 if hw else 128
 if hw:hw.reset_stats()
 for pos in range(0,len(ids),batch):
  if pos%512==0:status('evaluating',route=name or 'validation',seen=pos,total=len(ids))
  ix=ids[pos:pos+batch]
  if hw:hw.context(ix,seed or CFG['validation_perturbation_seed'])
  zs.append(m(prep(x[ix])).cpu())
 z=torch.cat(zs);d=dict(n=len(ids),accuracy=float((z.argmax(1)==y[ids]).float().mean()),CE=float(F.cross_entropy(z,y[ids])),seconds=time.monotonic()-begin)
 if hw:d['hardware']=hw.metrics()
 if name:np.savez_compressed(R/(name+'.npz'),ids=ids,labels=y[ids].numpy(),logits=z.numpy(),perturbation_seed=seed if seed is not None else -1)
 return d

def freeze_bn(m):
 for op in m.modules():
  if isinstance(op,nn.BatchNorm2d):op.eval()

def train(name,m,x,y,ids,valid,epochs,teacher=None,hw=None):
 floating=name=='floating';lr=CFG['floating_lr'] if floating else CFG['qat_lr'] if name=='W4A4' else CFG['hardware_lr']
 v=evaluate(m,x,y,valid,hw);best=(v['accuracy'],-v['CE']);bestval=v;bestep=0;history=[]
 torch.save(dict(model=m.state_dict(),epoch=0,validation=v),R/(name+'_best.pt'))
 if floating:opt=torch.optim.SGD(m.parameters(),lr=lr,momentum=.9,nesterov=True,weight_decay=5e-4)
 else:
  regular=[];scales=[]
  for n,p in m.named_parameters():(scales if 'scale' in n else regular).append(p)
  opt=torch.optim.AdamW([dict(params=regular,weight_decay=.01),dict(params=scales,weight_decay=0.)],lr=lr)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=CFG['floating_lr_min'] if floating else lr*.1)
 batch=CFG['floating_batch'] if floating else CFG['qat_batch'] if name=='W4A4' else CFG['hardware_batch']
 if teacher:
  teacher.eval()
  for p in teacher.parameters():p.requires_grad_(False)
 for epoch in range(1,epochs+1):
  phase=100 if floating else 200 if name=='W4A4' else 300
  torch.manual_seed(CFG['seed']+phase+epoch);order=ids[torch.randperm(len(ids)).numpy()];m.train()
  if not floating:freeze_bn(m)
  if hw:hw.reset_stats()
  correct=0;loss_sum=0.;begin=time.monotonic()
  for pos in range(0,len(order),batch):
   if pos%(batch*16)==0:status(name,epoch=epoch,epochs=epochs,seen=pos,total=len(ids),seconds=time.monotonic()-begin)
   ix=order[pos:pos+batch];a=prep(x[ix],True);labels=y[ix].cuda()
   if hw:hw.context(ix,CFG['seed']+1000+epoch)
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=floating):z=m(a);loss=F.cross_entropy(z,labels)
   if teacher:
    with torch.no_grad():target=teacher(a)
    loss=loss+2.*F.kl_div(F.log_softmax(z/2,1),F.softmax(target/2,1),reduction='batchmean')
   assert torch.isfinite(loss);loss.backward()
   if not floating:torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   opt.step()
   with torch.no_grad():
    if not floating:
     for op in m.modules():
      if isinstance(op,QuantOp):op.activation_scale.clamp_(min=1e-8);op.weight_scale.clamp_(min=1e-8)
   loss_sum+=float(loss.detach())*len(ix);correct+=int((z.detach().argmax(1)==labels).sum())
  scheduler.step();v=evaluate(m,x,y,valid,hw);score=(v['accuracy'],-v['CE'])
  history.append(dict(epoch=epoch,validation=v,train_accuracy=correct/len(ids),loss=loss_sum/len(ids),seconds=time.monotonic()-begin,lr=opt.param_groups[0]['lr']))
  if score>best:best=score;bestval=v;bestep=epoch;torch.save(dict(model=m.state_dict(),epoch=epoch,validation=v),R/(name+'_best.pt'))
  torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch),R/(name+'_last.pt'))
  save(name+'_history.json',history);RESULTS['stages'][name]=dict(validation=bestval,best_epoch=bestep,epochs_finished=epoch);report('running')
 m.load_state_dict(torch.load(R/(name+'_best.pt'),weights_only=True)['model']);m.eval();RESULTS['stages'][name]['checkpoint_sha256']=sha(R/(name+'_best.pt'));return m

def calibrate(m,x,ids):
 m.eval();samples={};hooks=[]
 for name,op in m.named_modules():
  if isinstance(op,(nn.Conv2d,nn.Linear)):
   def hook(op,args,n=name):
    a=args[0].detach().abs().flatten();samples.setdefault(n,[]).append(a[::max(1,a.numel()//4096)][:4096].cpu())
   hooks.append(op.register_forward_pre_hook(hook))
 with torch.no_grad():
  for i in range(0,len(ids),8):m(prep(x[ids[i:i+8]]))
 for h in hooks:h.remove()
 return {n:max(float(torch.quantile(torch.cat(v),.999))/7,1e-8) for n,v in samples.items()}

def exact_check(m,x,ids):
 hw=Hardware('digital');attach(m,hw);hw.context(ids[:1],0);m.eval()
 with torch.no_grad():
  inp=prep(x[ids[:1]]);a=m(inp);hw.mode='exact';b=m(inp)
 torch.testing.assert_close(a,b,rtol=0,atol=0)
 for op in m.modules():
  if isinstance(op,QuantOp):op.hw=None
 return {'exact_CIM_vs_digital_max_logit_error':float((a-b).abs().max())}

def gate(reason):RESULTS['reason']=reason;report('validation_gate');save('status.json',dict(state='validation_gate',reason=reason,utc=now()))
def main():
 global START
 assert json.loads((R/'preflight.json').read_text())['status']=='PASS'
 runtime();torch.manual_seed(CFG['seed']);START=time.monotonic()
 with (R/'RUN_STARTED.json').open('x') as f:json.dump(dict(utc=now(),pid=os.getpid()),f)
 save('protocol.json',CFG)
 sources=list(R.glob('*.py'))+list(R.glob('*.cu'))+[BASE/'model.py',BASE/'data.py',BASE/'split.json']
 manifest={str(p):sha(p) for p in sources};save('manifest.json',dict(sources=manifest,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
 x,y=read_data('train');split=create_split();ids=np.array(split['train']);valid=np.array(split['validation'])
 assert len(ids)==44976 and len(valid)==5000 and not set(ids)&set(valid)
 qids=subset(y,ids,100,CFG['seed']);hids=subset(y,ids,20,CFG['seed']);hval=subset(y,valid,5,CFG['seed']);calids=subset(y,ids,2,CFG['seed'])
 save('split.json',split);save('pilot_split.json',dict(qat=qids.tolist(),hardware_train=hids.tolist(),hardware_validation=hval.tolist(),calibration=calids.tolist()))
 m=train('floating',WRN().cuda(),x,y,ids,valid,CFG['floating_epochs'])
 if RESULTS['stages']['floating']['validation']['accuracy']<CFG['minimum_floating_validation_accuracy']:gate('Floating validation below predeclared 74%; no hardware tests.');return
 fp=copy.deepcopy(m).eval();scales=calibrate(m,x,calids);save('calibration.json',dict(source='train only',ids=calids.tolist(),scales=scales))
 q=quantize(copy.deepcopy(m),scales);q=train('W4A4',q,x,y,qids,valid,CFG['qat_epochs'],fp)
 gap=100*(RESULTS['stages']['floating']['validation']['accuracy']-RESULTS['stages']['W4A4']['validation']['accuracy'])
 if gap>CFG['maximum_QAT_gap_pp']:gate(f'Digital quantization gap {gap:.2f} pp exceeds 5 pp gate; no hardware test.');return
 RESULTS['initial_exact_check']=exact_check(q,x,qids)
 teacher=copy.deepcopy(q).eval();models={}
 for name,mode in [('adc_control','adc'),('sa_skip5','sa')]:
  hw=Hardware(mode);m=copy.deepcopy(q);attach(m,hw);models[name]=train(name,m,x,y,hids,hval,CFG['hardware_epochs'],teacher,hw)
 save('SELECTION_FROZEN.json',dict(utc=now(),stages=RESULTS['stages']))
 RESULTS['final_exact_check']=exact_check(models['sa_skip5'],x,qids)
 tx,ty=read_data('test');test=np.array(json.loads((BASE/'test_ids.json').read_text()));save('test_ids.json',test.tolist());RESULTS['test']={}
 for name,m,mode in [('floating',fp,None),('W4A4',q,None),('native_ADC',q,'adc'),('disturbance_only',q,'disturbance'),('direct_skip5',q,'sa'),('ordinary_ADC',models['adc_control'],'adc'),('ordinary_skip5',models['adc_control'],'sa'),('aware_skip5',models['sa_skip5'],'sa')]:
  hw=Hardware(mode) if mode else None
  if hw:attach(m,hw)
  RESULTS['test'][name]=evaluate(m,tx,ty,test,hw,'test_'+name,CFG['test_perturbation_seed']);report('running')
 RESULTS['paired']={};b=np.load(R/'test_aware_skip5.npz');cb=b['logits'].argmax(1)==b['labels']
 for name in ['direct_skip5','ordinary_skip5']:
  a=np.load(R/('test_'+name+'.npz'));assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels']);ca=a['logits'].argmax(1)==a['labels']
  RESULTS['paired'][name]=dict(gain_pp=float((cb.astype(float)-ca).mean()*100),wrong_to_correct=int((~ca&cb).sum()),correct_to_wrong=int((ca&~cb).sum()))
 for path,h in manifest.items():assert sha(Path(path))==h,path
 RESULTS['elapsed_seconds']=time.monotonic()-START;report('complete');save('status.json',dict(state='complete',utc=now()))
 (R/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(R.iterdir()) if p.is_file() and p.name not in ['SHA256SUMS','pipeline.log']))
if __name__=='__main__':
 if (R/'RUN_STARTED.json').exists():raise SystemExit('Existing run preserved')
 try:main()
 except Exception:
  RESULTS['reason']=traceback.format_exc();report('failed');save('status.json',dict(state='failed',utc=now(),reason=RESULTS['reason']));raise
