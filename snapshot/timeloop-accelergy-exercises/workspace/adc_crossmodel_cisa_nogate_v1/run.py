"""Bounded replication of the WRN no-gate matrix for two frozen W4A4 parents."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MAX_JOBS', '2')
import argparse, copy, datetime, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import hardware as hm
from hardware import Hardware
from models import build, attach, detach, prep, runtime, freeze_bn, TARGETS
from deit_data import read_data, subset
from oracle import oracle

R = Path(__file__).resolve().parent
SEED = 20260913
POINTS = [(target, skip) for target in (20,40) for skip in (4,5,6)]
BATCH = 4
EPOCHS = 20
OUT = None
STATE = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, data, root=None):
    p = (root or OUT)/name
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(p)

def status(stage, **kwargs):
    d = dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs)
    save('status.json', d)
    print(json.dumps(d), flush=True)

def report():
    save('results.json', STATE)
    s = '# New-SA cross-model replication: '+STATE['model']+'\n\n'
    s += 'W4A4, ADC8, 0–1.2 V, no data-dependent zero gating; original three mapped layers only.\n\n'
    s += '| Route | Accuracy | SAR comparison saving | Conversions |\n|---|---:|---:|---:|\n'
    for name, v in STATE.get('test', {}).items():
        h = v.get('hardware')
        saving = f"{100*h['comparison_saving']:.3f}%" if h else '—'
        s += f"| {name} | {100*v['accuracy']:.2f}% | {saving} | {h['conversions'] if h else '—'} |\n"
    s += '\nAll six operating points retained; no test-based selection. Historical test subset, not fresh blind data. '
    s += 'Deterministic B schematic residual interpolation with grid-edge clamp; no device MC or total-system net-energy claim.\n'
    (OUT/'REPORT.md').write_text(s)

def compile_kernel():
    (R/'build').mkdir(exist_ok=True)
    hm.EXT = load('crossmodel_cisa_nogate_v1', sources=[str(R/'events_cisa.cu')],
                  build_directory=str(R/'build'), extra_cuda_cflags=['-O3','--fmad=false'],
                  extra_cflags=['-O3'], verbose=False)

def circuit_preflight():
    rng = np.random.default_rng(SEED)
    raw = rng.integers(0,50,(2,4,2,3,128)).astype(np.float32)
    raw[:,2] = 0
    raw[..., :8] = [0,0,3,4,8,9,192,384]
    em = np.zeros((2,4,3),np.uint8)
    de = np.zeros((2,2,128),np.uint8); de[:,1,96:] = 1
    ids = torch.tensor([1,2,3],device='cuda')
    inputs = [torch.from_numpy(a).cuda() for a in (raw,em,de)]
    checks = []
    for target, skip in POINTS:
        for mode,dist in [(0,False),(1,False),(2,True),(3,False),(3,True)]:
            h = Hardware(target=target,skip=skip,disturbance=dist)
            out, stats, tr = hm.EXT.convert(*inputs,ids,1,0,0,mode,skip,h.theta,0.,True,h.table,dist,h.lower,h.upper)
            gold, ss, tt = oracle(raw,em,de,mode,h.table.cpu().numpy(),dist,h.theta,h.lower,h.upper,skip)
            assert np.array_equal(out.cpu().numpy(),gold)
            assert np.array_equal(stats.cpu().numpy(),ss)
            np.testing.assert_allclose(tr.cpu().numpy(),tt,rtol=0,atol=1e-11,equal_nan=True)
            if mode:
                assert ss[2] == 2*4*3*(128+96)
                assert ss[5] == 8*ss[2]-skip*ss[4]
            checks.append([target,skip,mode,dist])
    return dict(status='PASS',checks=checks,ordered_codes_counts_and_held_feedback_match=True)

def data_split(kind, x, y):
    excluded = set(json.loads((R/'input/deit_split.json').read_text())['excluded_train_ids'])
    if kind == 'deit':
        old = json.loads((R/'input/deit_pilot_split.json').read_text())
        train = np.array(old['hardware_train']); valid = np.array(old['hardware_validation'])
        test = np.array(json.loads((R/'input/deit_test_ids.json').read_text()))
    else:
        old = json.loads((R/'input/resnet_parent_split.json').read_text())
        train = subset(y, sorted(set(old['train'])-excluded),20,SEED)
        valid = subset(y, sorted(set(old['validation'])-excluded),5,SEED+1)
        test = np.array(json.loads((R/'input/resnet_old_hardware_split.json').read_text())['test'])
    assert len(train)==2000 and len(valid)==500 and len(test)==1000
    assert not set(train)&set(valid) and not (set(train)|set(valid))&excluded
    hashes = lambda ids: {hashlib.sha256(x[i].numpy().tobytes()).hexdigest() for i in ids}
    assert not hashes(train)&hashes(valid)
    return dict(train=train.tolist(),validation=valid.tolist(),test=test.tolist(),
                excluded_train_ids=sorted(excluded),training_validation_content_disjoint=True)

@torch.no_grad()
def exact_check(kind, m, x, ids):
    detach(m,kind); m.eval(); a = prep(x[ids[:2]],kind)
    z = m(a)
    h = Hardware('exact',disturbance=False); attach(m,h,kind); h.context(ids[:2],0)
    other = m(a)
    torch.testing.assert_close(z,other,rtol=0,atol=0)
    detach(m,kind)
    # Independent integer tests cross 128-row / 64-output boundaries and bit sign.
    class Op: pass
    op=Op();op.isconv=False;op.name='fixture'
    if kind=='resnet': op.low=0
    qx=torch.randint(0 if kind=='resnet' else -7,16 if kind=='resnet' else 8,(2,3,145),device='cuda').float()
    qw=torch.randint(-7,8,(70,145),device='cuda').float()
    h.context([0,1],0)
    torch.testing.assert_close(h.accumulator(op,qx,qw),F.linear(qx,qw).to(torch.float64 if kind=='resnet' else torch.float32),rtol=0,atol=0)
    return dict(status='PASS',max_logit_difference=float((z-other).abs().max()),
                activation='unsigned A[0,15]' if kind=='resnet' else 'signed A[-7,7]',
                tile_boundary_integer_dot_products_identical=True)

@torch.no_grad()
def evaluate(kind,m,x,y,ids,hw=None,name=None):
    m.eval(); attach(m,hw,kind) if hw else detach(m,kind)
    if hw: hw.reset_stats()
    chunks=[];begin=time.monotonic()
    for pos in range(0,len(ids),BATCH):
        ix=ids[pos:pos+BATCH]
        if hw: hw.context(ix,20261010)
        chunks.append(m(prep(x[ix],kind)).cpu())
    z=torch.cat(chunks); correct=int((z.argmax(1)==y[ids]).sum())
    assert torch.isfinite(z).all()
    result=dict(n=len(ids),correct=correct,accuracy=correct/len(ids),CE=float(F.cross_entropy(z,y[ids])),seconds=time.monotonic()-begin)
    if hw:
        result['hardware']=hw.metrics();h=result['hardware']
        assert h['gated']==0 and h['conversions']==h['requests']
        assert h['comparisons']==8*h['conversions']-hw.skip*h['near']
    if name: np.savez_compressed(OUT/(name+'.npz'),ids=ids,labels=y[ids].numpy(),logits=z.numpy())
    return result

def train(kind,name,m,teacher,hw,x,y,ids,valid):
    attach(m,hw,kind)
    initial=evaluate(kind,m,x,y,valid,hw)
    best=(initial['accuracy'],-initial['CE']);bestep=0;bestv=initial;history=[]
    torch.save(dict(model=m.state_dict(),epoch=0,validation=initial),OUT/(name+'_best.pt'))
    regular=[];scales=[]
    for n,p in m.named_parameters(): (scales if 'scale' in n else regular).append(p)
    opt=torch.optim.AdamW([dict(params=regular,weight_decay=.01),dict(params=scales,weight_decay=0.)],lr=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    bnbuf={n:v.clone() for n,v in m.state_dict().items() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
    for epoch in range(1,EPOCHS+1):
        torch.manual_seed(SEED+300+epoch)
        order=ids[torch.randperm(len(ids)).numpy()]
        m.train();freeze_bn(m);hw.reset_stats();begin=time.monotonic();loss_sum=0;correct=0
        for pos in range(0,len(order),BATCH):
            ix=order[pos:pos+BATCH];a=prep(x[ix],kind,True);labels=y[ix].cuda()
            hw.context(ix,SEED+1000+epoch);opt.zero_grad(set_to_none=True)
            with torch.no_grad(): target=teacher(a)
            z=m(a);loss=F.cross_entropy(z,labels)+2*F.kl_div(F.log_softmax(z/2,1),F.softmax(target/2,1),reduction='batchmean')
            assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for n,p in m.named_parameters():
                    if n.endswith(('activation_scale','weight_scale')): p.clamp_(min=1e-8)
            loss_sum+=float(loss.detach())*len(ix);correct+=int((z.detach().argmax(1)==labels).sum())
            if pos%512==0: status('training',model=kind,branch=name,epoch=epoch,epochs=EPOCHS,seen=pos+len(ix),total=len(ids),seconds=time.monotonic()-begin)
        train_hw=hw.metrics();sched.step()
        assert all(torch.equal(v,m.state_dict()[n]) for n,v in bnbuf.items())
        val=evaluate(kind,m,x,y,valid,hw);score=(val['accuracy'],-val['CE'])
        row=dict(epoch=epoch,validation=val,training_accuracy=correct/len(ids),loss=loss_sum/len(ids),training_hardware=train_hw,seconds=time.monotonic()-begin)
        history.append(row)
        if score>best:
            best=score;bestep=epoch;bestv=val
            torch.save(dict(model=m.state_dict(),epoch=epoch,validation=val),OUT/(name+'_best.pt'))
        torch.save(dict(model=m.state_dict(),epoch=epoch,optimizer=opt.state_dict(),scheduler=sched.state_dict()),OUT/(name+'_last.pt'))
        save(name+'_history.json',history)
        STATE['stages'][name]=dict(epochs_finished=epoch,best_epoch=bestep,validation=bestv)
        report();status('epoch_complete',model=kind,branch=name,epoch=epoch,validation_accuracy=val['accuracy'],best_accuracy=bestv['accuracy'])
    return m

def main(kind,preflight_only):
    global OUT,STATE
    OUT=R/kind;OUT.mkdir(exist_ok=True)
    runtime();torch.manual_seed(SEED)
    STATE=dict(model=kind,status='running',stages={},test={})
    compile_kernel()
    circuit=circuit_preflight()
    x,y=read_data('train');split=data_split(kind,x,y)
    ids=np.array(split['train']);valid=np.array(split['validation'])
    q=build(kind)
    preflight=exact_check(kind,q,x,ids)
    # Verify old quantizer implementation remains unchanged for digital inference.
    h=Hardware(target=20,skip=4);m=copy.deepcopy(q);attach(m,h,kind);h.context(ids[:2],0)
    m.train();freeze_bn(m);begin=time.monotonic()
    loss=F.cross_entropy(m(prep(x[ids[:2]],kind)),y[ids[:2]].cuda());loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    preflight.update(circuit=circuit,hardware_backward_finite=True,smoke_step_seconds=time.monotonic()-begin)
    save('preflight.json',preflight);print('PREFLIGHT_PASS',kind,json.dumps(preflight),flush=True)
    del m,h;torch.cuda.empty_cache()
    if preflight_only:return
    with (OUT/'RUN_STARTED.json').open('x') as f: json.dump(dict(pid=os.getpid(),start=time.time()),f)
    sources=[p for p in R.iterdir() if p.suffix in ('.py','.cu','.bin','.json')]+list((R/'input').iterdir())
    manifest={str(p.relative_to(R)):sha(p) for p in sources if p.is_file()}
    save('manifest.json',dict(sources=manifest,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),seed=SEED,hardware_layers=TARGETS[kind]))
    save('split.json',split);save('protocol.json',json.loads((R/'FROZEN_PROTOCOL.json').read_text()))
    teacher=copy.deepcopy(q).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    branches=[('ordinary_ADC',None)]+[(f'aware_{t}_skip{s}',(t,s)) for t,s in POINTS]
    for name,point in branches:
        hw=Hardware('adc',disturbance=False) if point is None else Hardware(target=point[0],skip=point[1])
        m=train(kind,name,copy.deepcopy(q),teacher,hw,x,y,ids,valid)
        del m,hw;torch.cuda.empty_cache()
    save('SELECTION_FROZEN.json',STATE['stages'])
    tx,ty=read_data('test');test=np.array(split['test'])
    def ev(name,m,hw=None):
        status('testing',model=kind,route=name)
        v=evaluate(kind,m,tx,ty,test,hw,'test_'+name)
        if hw and 'native_ADC' in STATE['test']:
            assert v['hardware']['requests']==STATE['test']['native_ADC']['hardware']['requests']
        STATE['test'][name]=v;report()
    ev('digital_W4A4',q)
    ev('native_ADC',q,Hardware('adc',disturbance=False))
    control=build(kind);control.load_state_dict(torch.load(OUT/'ordinary_ADC_best.pt',weights_only=True)['model'])
    ev('ordinary_native_ADC',control,Hardware('adc',disturbance=False))
    for t in (20,40):ev(f'disturbance_only_{t}',q,Hardware('disturbance',target=t))
    for t,s in POINTS:
        ev(f'decision_only_{t}_skip{s}',q,Hardware(target=t,skip=s,disturbance=False))
        ev(f'direct_{t}_skip{s}',q,Hardware(target=t,skip=s))
        ev(f'ordinary_{t}_skip{s}',control,Hardware(target=t,skip=s))
        m=build(kind);m.load_state_dict(torch.load(OUT/f'aware_{t}_skip{s}_best.pt',weights_only=True)['model'])
        ev(f'aware_{t}_skip{s}',m,Hardware(target=t,skip=s));del m;torch.cuda.empty_cache()
    assert all(sha(R/p)==h for p,h in manifest.items())
    STATE['status']='complete';report();status('complete',model=kind)
    (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['resnet','deit']);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
    try:main(a.model,a.preflight_only)
    except Exception:
        if OUT is not None:
            save('failure_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc()))
            status('failed',model=a.model,reason=traceback.format_exc())
        raise
