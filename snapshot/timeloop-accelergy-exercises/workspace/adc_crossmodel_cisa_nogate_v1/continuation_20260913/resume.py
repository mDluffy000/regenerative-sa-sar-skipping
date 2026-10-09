"""Continue the paused fixed protocols without changing original run artifacts."""
from pathlib import Path
import os,sys,json,shutil,copy,time,hashlib,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
D=Path(__file__).resolve().parent;R=D.parent
sys.path.insert(0,str(R))
import torch,numpy as np
import run as t
from models import build,attach,detach,runtime,TARGETS
from hardware import Hardware
from deit_data import read_data
exec(compile((D/'resume_training.py').read_text(),str(D/'resume_training.py'),'exec'),t.__dict__)

def verify_sources(kind):
    manifest=json.loads((R/kind/'manifest.json').read_text())
    for name,h in manifest['sources'].items():assert t.sha(R/name)==h,name
    return manifest

def init_output(job,old,kind,epochs):
    out=D/job;out.mkdir(exist_ok=False)
    for p in old.iterdir():
        if p.is_file():shutil.copy2(p,out/p.name)
    source=verify_sources(kind)
    t.OUT=out;t.EPOCHS=epochs
    t.STATE=json.loads((old/'results.json').read_text())
    assert not t.STATE.get('test'),'Paused run unexpectedly already has tests'
    t.STATE.update(status='running',test={})
    t.save('CONTINUATION.json',dict(start=time.time(),pid=os.getpid(),parent=str(old),parent_source_manifest_verified=True,source_torch=source['torch'],current_torch=torch.__version__,epochs=epochs,partial_epoch_replayed=True))
    return out

def test_primary(kind,q,out,split):
    t.save('SELECTION_FROZEN.json',t.STATE['stages'])
    tx,ty=read_data('test');ids=np.array(split['test'])
    def ev(name,m,hw=None):
        t.status('testing',model=kind,route=name)
        v=t.evaluate(kind,m,tx,ty,ids,hw,'test_'+name)
        if hw and 'native_ADC' in t.STATE['test']:
            assert v['hardware']['requests']==t.STATE['test']['native_ADC']['hardware']['requests']
        t.STATE['test'][name]=v;t.report()
    ev('digital_W4A4',q)
    ev('native_ADC',q,Hardware('adc',disturbance=False))
    control=build(kind);control.load_state_dict(torch.load(out/'ordinary_ADC_best.pt',weights_only=True)['model'])
    ev('ordinary_native_ADC',control,Hardware('adc',disturbance=False))
    for target in (20,40):ev(f'disturbance_only_{target}',q,Hardware('disturbance',target=target))
    for target,skip in t.POINTS:
        ev(f'decision_only_{target}_skip{skip}',q,Hardware(target=target,skip=skip,disturbance=False))
        ev(f'direct_{target}_skip{skip}',q,Hardware(target=target,skip=skip))
        ev(f'ordinary_{target}_skip{skip}',control,Hardware(target=target,skip=skip))
        m=build(kind);m.load_state_dict(torch.load(out/f'aware_{target}_skip{skip}_best.pt',weights_only=True)['model'])
        ev(f'aware_{target}_skip{skip}',m,Hardware(target=target,skip=skip));del m;torch.cuda.empty_cache()

def primary(kind):
    out=init_output(kind,R/kind,kind,20)
    split=json.loads((out/'split.json').read_text());ids=np.array(split['train']);valid=np.array(split['validation'])
    x,y=read_data('train');q=build(kind);teacher=copy.deepcopy(q).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    t.STATE['continuation_exact_checks']={}
    for name,point in [('ordinary_ADC',None)]+[(f'aware_{a}_skip{b}',(a,b)) for a,b in t.POINTS]:
        h=Hardware('adc',disturbance=False) if point is None else Hardware(target=point[0],skip=point[1])
        m=t.train_resume(kind,name,copy.deepcopy(q),teacher,h,x,y,ids,valid)
        m.load_state_dict(torch.load(out/(name+'_best.pt'),weights_only=True)['model'])
        t.STATE['continuation_exact_checks'][name]=t.exact_check(kind,m,x,ids)
        del m,h;torch.cuda.empty_cache()
    test_primary(kind,q,out,split)
    return out

def bn_diagnostic():
    kind='resnet';old=R/'diagnostics/bn_refresh_finetune'
    out=init_output('bn_refresh_finetune',old,kind,5)
    split=json.loads((R/'resnet/split.json').read_text());ids=np.array(split['train']);valid=np.array(split['validation'])
    t.save('split.json',split)
    t.save('TEST_PROTOCOL.json',dict(protocol='Finish the three previously fixed 5-epoch branches, freeze all validation choices, evaluate each on the same historical 1000-image test subset as primary ResNet. Report BN-only and BN+finetuning for all three routes; do not select routes or tune using test.',primary_matrix_unchanged=True,exploratory_adaptation=True))
    x,y=read_data('train');teacher=build(kind).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    routes=[('native_ADC',None,4),('SA20_skip4',20,4),('SA40_skip6',40,6)]
    for name,target,skip in routes:
        m=build(kind);m.load_state_dict(torch.load(R/'diagnostics/bn_refresh'/(name+'.pt'),weights_only=True)['model'])
        h=Hardware('adc',disturbance=False) if target is None else Hardware(target=target,skip=skip)
        t.train_resume(kind,name,m,teacher,h,x,y,ids,valid)
        m.load_state_dict(torch.load(out/(name+'_best.pt'),weights_only=True)['model'])
        t.save(name+'_selected_validation.json',t.evaluate(kind,m,x,y,valid,h,name+'_selected_validation'))
        t.save(name+'_exact_check.json',t.exact_check(kind,m,x,ids));del m,h;torch.cuda.empty_cache()
    t.save('SELECTION_FROZEN.json',t.STATE['stages'])
    tx,ty=read_data('test');test=np.array(split['test'])
    for name,target,skip in routes:
        for stage in ['BN_only','BN_finetuned']:
            path=R/'diagnostics/bn_refresh'/(name+'.pt') if stage=='BN_only' else out/(name+'_best.pt')
            m=build(kind);m.load_state_dict(torch.load(path,weights_only=True)['model'])
            h=Hardware('adc',disturbance=False) if target is None else Hardware(target=target,skip=skip)
            route=stage+'_'+name;t.status('testing',model=kind,route=route)
            t.STATE['test'][route]=t.evaluate(kind,m,tx,ty,test,h,'test_'+route);t.report()
            del m,h;torch.cuda.empty_cache()
    return out

def main(job):
    runtime();torch.manual_seed(t.SEED);t.compile_kernel()
    out=bn_diagnostic() if job=='bn' else primary(job)
    verify_sources('resnet' if job=='bn' else job)
    t.STATE['status']='complete';t.report();t.status('complete',job=job)
    hashes={str(p.relative_to(out)):t.sha(p) for p in out.iterdir() if p.is_file() and p.name!='FINAL_SHA256.json'}
    t.save('FINAL_SHA256.json',hashes)

if __name__=='__main__':
    job=sys.argv[1];assert job in ('resnet','deit','bn')
    try:main(job)
    except Exception:
        if t.OUT is not None:t.status('failed',job=job,reason=traceback.format_exc())
        raise
