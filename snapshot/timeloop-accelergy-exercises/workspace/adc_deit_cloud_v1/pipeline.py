"""Frozen first DeiT pilot. Train-only calibration; test deferred until stage gates pass.

Three early/middle/late MLP output projections carry ADC/SA error. All 50 fixed
weight operators are W4A4. This is not full-transformer analog coverage.
"""
from pathlib import Path
import copy,datetime,hashlib,json,os,time,traceback
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model import pretrained,quantize,QuantOp,configure_runtime
from hardware import Hardware,attach,PILOT_TARGETS
from data import read_data,preprocess,create_split,subset
from preflight import calibrate
R=Path(__file__).resolve().parent;W=R.parent
CFG=dict(seed=20260910,threads=8,head_epochs=30,head_lr=.003,
 fp_epochs=3,fp_training_images=10000,fp_lr=.00005,
 qat_epochs=5,qat_training_images=10000,qat_lr=.00005,
 aware_epochs=3,aware_training_images=2000,aware_lr=.00001,
 training_batch=8,hardware_batch=4,validation_images=1000,hardware_validation_images=500,
 pilot_test_images=1000,evaluation_perturbation_seed=20261010,validation_perturbation_seed=20261011,
 minimum_FP_validation_accuracy=.60,maximum_QAT_gap_pp=5.,
 checkpoint_selection='highest validation accuracy then lowest CE, epoch0 eligible; never test',
 quantization='all50 fixed-weight operators W[-7,7], signed A[-7,7]; learned scales',
 hardware_targets=PILOT_TARGETS,skip_bits=5,SA_window_LSB=8.,synthetic_held_error_sigma_LSB=.03,
 physical_model='ASSUMED independent Gaussian held-voltage error; wide common-mode behavior abstracted; no physical validation claim',
 architecture='128 rows x 128 columns; 2 slices/output; one sequential ADC service domain per array read; reset at each array/bit/token/reduction tile',
 unquantized='LayerNorm, softmax, GELU, residual boundaries, dynamic QK^T and attention*V stay FP32 digital',
 teacher='frozen digital FP task model for QAT; frozen digital W4A4 parent for aware and ADC-only arms',
 maximum_wall_seconds=5400,full_test_repeated_history='Official test previously used in project; this new model is not used to tune against test.',
 not_device_MC=True,not_full_analog_Transformer=True)
START=None;RESULTS={'config':CFG,'stages':{},'status':'running'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def status(stage,**kw):
 if START is not None and time.monotonic()-START>CFG['maximum_wall_seconds']:raise TimeoutError('90-minute cloud pilot runtime limit reached; completed checkpoints preserved')
 save('status.json',dict(state='running',stage=stage,utc=now(),**kw));print(stage,json.dumps(kw),flush=True)
def summarize(state):
 RESULTS['status']=state;save('results.json',RESULTS)
 s='# DeiT-Tiny / CIFAR-100 行为级试验\n\n状态：'+state+'。\n\n'
 s+='官方 ImageNet 预训练权重迁移到 CIFAR-100，输入维持224×224。全部50个固定权重算子W4A4；当前ADC/SA只覆盖第0/5/11块的MLP输出投影。动态注意力乘法、归一化、非线性和残差边界保留数字FP32。不能称为全Transformer模拟计算验证。\n\n'
 s+='使用预先声明的合成行为模型：SA窗口8 LSB、保持误差独立高斯σ=0.03 LSB。不是B实测拟合，不使用旧test输入衍生残差库，不宣称为已实现SA规格。训练/校准使用官方train，已知重复内容按训练侧排除；官方test历史已使用，因此不声称全新盲测。\n\n'
 s+='| 阶段 | 验证精度 | 选中轮数 |\n|---|---:|---:|\n'
 for name,v in RESULTS['stages'].items():
  if 'validation' in v:s+=f"| {name} | {100*v['validation']['accuracy']:.2f}% | {v.get('best_epoch','—')} |\n"
 if RESULTS.get('test'):
  s+='\n冻结模型后统一测试，所有路线同一1000图、固定扰动种子；下列精度不与不同数据集/样本数的历史精度混算。\n\n| 路线 | 准确率 | ADC/SAR比较次数 |\n|---|---:|---:|\n'
  for name,v in RESULTS['test'].items():s+=f"| {name} | {100*v['accuracy']:.2f}% | {v.get('hardware',{}).get('comparisons','—')} |\n"
 if 'paired' in RESULTS:s+='\n训练前后和普通微调对照的配对结果：\n\n```json\n'+json.dumps(RESULTS['paired'],indent=2)+'\n```\n'
 if RESULTS.get('reason'):s+='\n停止原因：'+RESULTS['reason']+'\n'
 s+='\n比较次数是三个映射层内的计数，不是全系统节能、时延或吞吐。保留全部阶段、验证历史和epoch0候选；不根据测试结果选择checkpoint。运行配置在protocol.json，逐图输出在test_*.npz，数值检查和实测速度在preflight.json。\n'
 (R/'REPORT.md').write_text(s)
@torch.no_grad()
def evaluate(m,x,y,ids,name=None,hw=None,seed=None):
 m.eval();zs=[];n=len(ids);begin=time.monotonic()
 if hw is not None:hw.reset_stats()
 batch=CFG['hardware_batch'] if hw and hw.mode!='digital' else 16
 for pos in range(0,n,batch):
  if pos%512==0:status('evaluating',route=name or 'validation',seen=pos,total=n)
  ix=ids[pos:pos+batch]
  if hw is not None:hw.context(ix,seed or CFG['validation_perturbation_seed'])
  zs.append(m(preprocess(x[ix]).cuda()).cpu())
 z=torch.cat(zs);correct=int((z.argmax(1)==y[ids]).sum())
 d=dict(n=n,correct=correct,accuracy=correct/n,CE=float(F.cross_entropy(z,y[ids])),seconds=time.monotonic()-begin)
 if hw is not None:d['hardware']=hw.metrics()
 if name:np.savez_compressed(R/(name+'.npz'),ids=np.asarray(ids),labels=y[ids].numpy(),logits=z.numpy(),perturbation_seed=seed if seed is not None else -1)
 return d
def feature_probe(m,x,y,split,valid):
 ids=np.asarray(sorted(split['train']+split['validation']));m.eval();features=torch.zeros((len(x),192),device="cuda")
 begin=time.monotonic()
 with torch.inference_mode():
  for pos in range(0,len(ids),128):
   if pos%512==0:status('feature_cache',seen=pos,total=len(ids),seconds=time.monotonic()-begin)
   ix=ids[pos:pos+128];features[ix]=m.features(preprocess(x[ix]).cuda())
 np.savez_compressed(R/'train_feature_cache.npz',ids=ids,features=features[ids].cpu().numpy())
 m.head=nn.Linear(192,100).cuda();nn.init.trunc_normal_(m.head.weight,std=.02);nn.init.zeros_(m.head.bias)
 opt=torch.optim.AdamW(m.head.parameters(),lr=CFG['head_lr'],weight_decay=.01);train=np.asarray(split['train']);best=(-1.,-1e20);history=[]
 for epoch in range(1,CFG['head_epochs']+1):
  torch.manual_seed(CFG['seed']+epoch);order=train[torch.randperm(len(train)).numpy()]
  m.head.train()
  for pos in range(0,len(order),512):
   ix=order[pos:pos+512];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m.head(features[ix]),y[ix].cuda());loss.backward();opt.step()
  with torch.no_grad():
   m.head.eval();z=m.head(features[valid]).cpu();v={'accuracy':float((z.argmax(1)==y[valid]).float().mean()),'CE':float(F.cross_entropy(z,y[valid]))}
  history.append({'epoch':epoch,'validation':v});score=(v['accuracy'],-v['CE'])
  if score>best:best=score;torch.save({'model':m.state_dict(),'epoch':epoch,'validation':v},R/'head_probe_best.pt')
  status('head_probe',epoch=epoch,validation=v)
 save('head_probe_history.json',history);chosen=torch.load(R/'head_probe_best.pt',weights_only=True);m.load_state_dict(chosen['model'])
 RESULTS['stages']['head_probe']={'validation':chosen['validation'],'best_epoch':chosen['epoch']};summarize('running')
 return m
def train_phase(name,m,x,y,train,valid,epochs,lr,teacher=None,hw=None):
 v=evaluate(m,x,y,valid,hw=hw);best=(v['accuracy'],-v['CE']);best_epoch=0;best_validation=v;history=[]
 torch.save({'model':m.state_dict(),'epoch':0,'validation':v},R/(name+'_best.pt'))
 normal=[];scales=[]
 for n,p in m.named_parameters():(scales if 'scale' in n else normal).append(p)
 opt=torch.optim.AdamW([{'params':normal,'weight_decay':.01},{'params':scales,'weight_decay':0.}],lr=lr)
 schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=lr*.1)
 if teacher is not None:
  teacher.eval()
  for p in teacher.parameters():p.requires_grad_(False)
 for epoch in range(1,epochs+1):
  phase_seed={'FP32':100,'W4A4':200,'adc_control':300,'sa_skip5':300}[name]
  torch.manual_seed(CFG['seed']+phase_seed+epoch);order=train[torch.randperm(len(train)).numpy()];m.train();total=0.;correct=0;begin=time.monotonic()
  if hw:hw.reset_stats()
  batch=CFG['hardware_batch'] if hw else CFG['training_batch']
  for pos in range(0,len(order),batch):
   if pos%256==0:status(name,epoch=epoch,seen=pos,total=len(order),seconds=time.monotonic()-begin)
   ix=order[pos:pos+batch];a=preprocess(x[ix],True).cuda()
   if hw:hw.context(ix,CFG['seed']+1000+epoch)
   opt.zero_grad(set_to_none=True);z=m(a);loss=F.cross_entropy(z,y[ix].cuda())
   if teacher is not None:
    with torch.no_grad():target=teacher(a)
    loss=loss+2.*F.kl_div(F.log_softmax(z/2,1),F.softmax(target/2,1),reduction='batchmean')
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
   with torch.no_grad():
    for op in m.modules():
     if isinstance(op,QuantOp):op.activation_scale.clamp_(min=1e-8);op.weight_scale.clamp_(min=1e-8)
   total+=float(loss.detach())*len(ix);correct+=int((z.detach().argmax(1).cpu()==y[ix]).sum())
  schedule.step();v=evaluate(m,x,y,valid,hw=hw);score=(v['accuracy'],-v['CE'])
  history.append({'epoch':epoch,'validation':v,'loss':total/len(train),'train_accuracy':correct/len(train),'seconds':time.monotonic()-begin})
  if score>best:best=score;best_epoch=epoch;best_validation=v;torch.save({'model':m.state_dict(),'epoch':epoch,'validation':v},R/(name+'_best.pt'))
  torch.save({'model':m.state_dict(),'epoch':epoch,'optimizer':opt.state_dict()},R/(name+'_last.pt'));save(name+'_history.json',history)
  RESULTS['stages'][name]={'validation':best_validation,'best_epoch':best_epoch,'epochs_finished':epoch};summarize('running')
 chosen=torch.load(R/(name+'_best.pt'),weights_only=True);m.load_state_dict(chosen['model']);RESULTS['stages'][name]={'validation':chosen['validation'],'best_epoch':chosen['epoch'],'checkpoint_sha256':sha(R/(name+'_best.pt'))};summarize('running');return m
def gate(reason):
 RESULTS['reason']=reason;summarize('stopped_at_validation_gate');save('status.json',{'state':'validation_gate','reason':reason,'utc':now()})
def final_checks(parent,trained,x,ids):
 checks=[]
 for label,model in [('QAT_parent',parent),('SA_trained',trained)]:
  hw=Hardware('exact');attach(model,hw);inp=preprocess(x[ids[:1]]).cuda();hw.context(ids[:1],0);model.eval()
  with torch.no_grad():
   hw.mode='digital';a=model(inp);hw.mode='exact';b=model(inp)
  torch.testing.assert_close(a,b,rtol=0,atol=0)
  checks.append({'model':label,'exact_CIM_vs_digital_logit_max_error':float((a-b).abs().max())})
  for op in model.modules():
   if isinstance(op,QuantOp):op.hw=None
 return checks
def main():
 global START
 if (R/'RUN_STARTED.json').exists():raise RuntimeError('Existing run preserved')
 assert json.loads((R/'preflight.json').read_text())['status']=='PASS'
 assert json.loads((R/'cloud_checks.json').read_text())['status']=='PASS'
 configure_runtime();torch.manual_seed(CFG['seed']);torch.use_deterministic_algorithms(True);START=time.monotonic()
 with (R/'RUN_STARTED.json').open('x') as f:json.dump({'utc':now(),'pid':os.getpid()},f)
 save('protocol.json',CFG)
 sources=list(R.glob('*.py'))+[R/'events.cpp',R/'events.so',R/'split.json',R/'input/deit_tiny_patch16_224-a1311bcf.pth']
 manifest={'sources':{str(p.relative_to(R)):sha(p) for p in sources},'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'start_utc':now()};save('manifest.json',manifest)
 split=create_split();x,y=read_data('train');valid=subset(y,split['validation'],10,CFG['seed']);hvalid=subset(y,split['validation'],5,CFG['seed'])
 train=subset(y,split['train'],100,CFG['seed']);htrain=subset(y,split['train'],20,CFG['seed'])
 save('pilot_split.json',{'train':train.tolist(),'hardware_train':htrain.tolist(),'validation':valid.tolist(),'hardware_validation':hvalid.tolist()})
 m=feature_probe(pretrained().cuda(),x,y,split,valid)
 m=train_phase('FP32',m,x,y,train,valid,CFG['fp_epochs'],CFG['fp_lr'])
 if RESULTS['stages']['FP32']['validation']['accuracy']<CFG['minimum_FP_validation_accuracy']:gate('FP32 task baseline below predeclared 60% validation gate; no SA training or test evaluation.');return
 fp=copy.deepcopy(m).eval();calids=subset(y,split['train'],2,CFG['seed']);scales=calibrate(m.eval(),x,calids)
 save('qat_calibration.json',{'ids':calids.tolist(),'scales':scales,'source':'train only','signed_activation':True})
 q=quantize(copy.deepcopy(m),scales);q=train_phase('W4A4',q,x,y,train,valid,CFG['qat_epochs'],CFG['qat_lr'],teacher=fp)
 gap=100*(RESULTS['stages']['FP32']['validation']['accuracy']-RESULTS['stages']['W4A4']['validation']['accuracy'])
 if gap>CFG['maximum_QAT_gap_pp']:gate(f'Digital W4A4 validation gap {gap:.2f} pp exceeds frozen 5 pp gate; preserve stage, improve digital baseline before SA test.');return
 teacher=copy.deepcopy(q).eval();models={}
 for arm in ['adc_control','sa_skip5']:
  hw=Hardware('adc' if arm=='adc_control' else 'sa');m=copy.deepcopy(q);attach(m,hw)
  m=train_phase(arm,m,x,y,htrain,hvalid,CFG['aware_epochs'],CFG['aware_lr'],teacher=teacher,hw=hw);models[arm]=m
 # Only after all model/epoch decisions are frozen do we read official test.
 tx,ty=read_data('test');test=subset(ty,np.arange(len(ty)),10,20261012);save('test_ids.json',test.tolist())
 RESULTS['test']={};RESULTS['test']['FP32']=evaluate(fp,tx,ty,test,'test_FP32')
 RESULTS['test']['W4A4']=evaluate(q,tx,ty,test,'test_W4A4')
 routes=[('native_ADC',q,'adc'),('disturbance_only',q,'disturbance'),('direct_skip5',q,'sa'),('ordinary_ADC',models['adc_control'],'adc'),('ordinary_skip5',models['adc_control'],'sa'),('aware_skip5',models['sa_skip5'],'sa')]
 for name,m,mode in routes:
  hw=Hardware(mode);attach(m,hw);RESULTS['test'][name]=evaluate(m,tx,ty,test,'test_'+name,hw,CFG['evaluation_perturbation_seed']);summarize('running')
 RESULTS['paired']={}
 for baseline in ['direct_skip5','ordinary_skip5']:
  a=np.load(R/('test_'+baseline+'.npz'));b=np.load(R/'test_aware_skip5.npz');assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels'])
  ca=a['logits'].argmax(1)==a['labels'];cb=b['logits'].argmax(1)==b['labels'];RESULTS['paired'][baseline]={'gain_pp':float((cb.astype(float)-ca).mean()*100),'wrong_to_correct':int((~ca&cb).sum()),'correct_to_wrong':int((ca&~cb).sum())}
 RESULTS['final_exact_checks']=final_checks(q,models['sa_skip5'],x,train)
 for rel,h in manifest['sources'].items():assert sha(R/rel)==h,rel
 RESULTS['elapsed_seconds']=time.monotonic()-START;summarize('complete');save('status.json',{'state':'complete','utc':now()})
 (R/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(R.iterdir()) if p.is_file() and p.name not in ('SHA256SUMS','pipeline.log')))
if __name__=='__main__':
 try:main()
 except Exception:
  RESULTS['reason']=traceback.format_exc();summarize('failed');save('status.json',{'state':'failed','utc':now(),'error':RESULTS['reason']});raise
