from __future__ import annotations
import sys,json,pickle,hashlib,csv,time,platform,subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from run_experiment import load_model, TARGETS, CHECKPOINT, MODEL_REPO
from phase2_experiment import replace_module
from int4_core import *
SEED=20260907
DATA=Path('/home/workspace/bf16_fp_cim_reproduction/data/cifar-100-python')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2))
def writecsv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
def load_data(split,ids):
    with (DATA/split).open('rb') as f: raw=pickle.load(f,encoding='bytes')
    x=torch.from_numpy(raw[b'data'][ids].copy()).reshape(-1,3,32,32).float()/255
    mean=torch.tensor([.5071,.4867,.4408])[None,:,None,None]
    std=torch.tensor([.2675,.2565,.2761])[None,:,None,None]
    return (x-mean)/std,torch.tensor(np.array(raw[b'fine_labels'])[ids].copy())

class QuantOp(torch.nn.Module):
    def __init__(self,name,op,scale,mode,owner):
        super().__init__();self.name=name;self.op=op;self.sa=scale;self.mode=mode;self.owner=owner
        qw,sw=weight_quantize(op.weight.detach());self.register_buffer('qw',qw);self.register_buffer('sw',sw)
        self.signed=name=='conv1';self.low=-7 if self.signed else 0;self.high=7 if self.signed else 15
        self.stats=defaultdict(int)
    def forward(self,x):
        if not self.signed: assert float(x.min()) >= 0, f'negative unsigned activation: {self.name}'
        lo,hi=self.low*self.sa,self.high*self.sa
        self.stats['activation_values']+=x.numel();self.stats['activation_clipped']+=int(((x<lo)|(x>hi)).sum())
        self.stats['activation_lower_clipped']+=int((x<lo).sum());self.stats['activation_upper_clipped']+=int((x>hi).sum())
        qx=quantize(x,self.sa,self.low,self.high)
        isconv=isinstance(self.op,torch.nn.Conv2d)
        golden=integer_golden(qx,self.qw,self.op if isconv else None)
        acc=golden
        if self.name in TARGETS and self.mode in ('int4_cim_exact','int4_cim_adc8_no_reuse'):
            p=patches(qx,self.op)
            ex,ad,met,es=cim_reconstruct(p,self.qw.flatten(1),collect=self.owner.collect and self.mode=='int4_cim_adc8_no_reuse')
            ex=ex.transpose(1,2).reshape_as(golden);ad=ad.transpose(1,2).reshape_as(golden)
            assert torch.equal(ex,golden)
            for k,v in met.items():self.stats[k]+=v
            self.stats['integer_mismatch']+=int((ex!=golden).sum())
            self.stats['integer_values_checked']+=golden.numel()
            acc=ad if self.mode=='int4_cim_adc8_no_reuse' else ex
            if es:self.owner.trace[self.name]=(qx.clone(),self.qw.clone(),self.sa,self.sw.clone(),ex.clone(),ad.clone(),es)
        if isconv: out=dequantize(acc,self.sa,self.sw,self.op.bias)
        else:
            out=(acc.double()*self.sa*self.sw.double()[None,:]).float()
            if self.op.bias is not None:out+=self.op.bias
        if self.owner.collect:
            self.owner.outputs[self.name]=out.clone()
            # Local error on same incoming tensor isolates this operator's quantization.
            fp=self.op(x)
            goldout=dequantize(golden,self.sa,self.sw,self.op.bias) if isconv else (golden.double()*self.sa*self.sw.double()[None,:]).float()+self.op.bias
            self.owner.local[self.name]=dict(fp_to_w4a4_mse=float(((goldout-fp).double()**2).mean()),
                finite_adc_local_mse=float(((out-goldout).double()**2).mean()),
                finite_adc_local_max=float((out-goldout).abs().max()))
            self.owner.qinputs[self.name]=qx.clone()
        return out

class State:
    def __init__(self):self.collect=False;self.outputs={};self.trace={};self.local={};self.qinputs={}
def build(scales,mode):
    model=load_model();state=State()
    for name,op in list(model.named_modules()):
        if isinstance(op,(torch.nn.Conv2d,torch.nn.Linear)):
            replace_module(model,name,QuantOp(name,op,scales[name],mode,state))
    return model,state

def calibrate():
    ids=np.random.default_rng(SEED).choice(50000,512,replace=False).tolist()
    x,y=load_data('train',ids);model=load_model();ranges={};handles=[]
    def hook(name,args):
        t=args[0];v=float(t.abs().max()) if name=='conv1' else float(t.max())
        if name!='conv1':assert float(t.min())>=0
        ranges[name]=max(ranges.get(name,0),v)
    for name,m in model.named_modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):
            handles.append(m.register_forward_pre_hook(lambda m,args,n=name:hook(n,args)))
    with torch.inference_mode():
        for s in range(0,len(x),64):model(x[s:s+64])
    for h in handles:h.remove()
    scales={n:max(v,1e-12)/(7 if n=='conv1' else 15) for n,v in ranges.items()}
    savej('calibration.json',dict(split='train',ids=ids,seed=SEED,method='FP-model per-input-tensor absolute max for first layer, max otherwise; frozen before test',ranges=ranges,scales=scales))
    print('calibration frozen: 512 train images,',len(scales),'operators',flush=True)
    return scales,ids

def export_trace(state):
    archive={};descriptors=[]
    for layer,(qx,qw,sa,sw,ex,ad,es) in state.trace.items():
        key=layer.replace('.','_')
        def put(n,v):archive[key+'__'+n]=v.numpy() if isinstance(v,torch.Tensor) else np.array(v)
        raw=torch.stack([e['raw'] for e in es]).to(torch.int16)
        put('raw_integer_psum',raw);put('analog_normalized_fs',raw.float()/ADC_FS)
        put('adc_code',torch.stack([e['code'] for e in es]))
        evidence=torch.stack([e['evidence'] for e in es]);put('zero_evidence_type',evidence);put('zero_preknown',evidence!=0)
        for n,f in [('row_chunk_id','row_chunk'),('activation_bit','bit'),('weight_slice','slice'),('sign_branch','branch'),('candidate_shift','shift')]:put(n,[e[f] for e in es])
        put('candidate_scale',sa*sw.double());put('input_q',qx.to(torch.int8));put('weight_q',qw.to(torch.int8))
        put('exact_integer_accumulator',ex);put('adc8_accumulator',ad)
        put('sample_id',[0,1]);put('layer_id',layer);put('run_id','int4_v1_20260907')
        put('adc_lane_id',0);put('bank_id',0);put('transfer_id','linear_u8_raw384_rne_v1')
        put('vfs_v',np.nan);put('analog_voltage_v',np.nan);put('voltage_mapping_id','normalized_only_v1')
        put('physical_voltage_mapping_status','unverified')
        descriptors.append(dict(layer=layer,prefix=key,shape=list(raw.shape),events=raw.numel(),axis_order=['event_group','sample','window','column']))
    np.savez_compressed(ROOT/'adc_trace_sample.npz',**archive)
    savej('trace_index.json',descriptors)

def evaluate(model,x,y,label,quant=False):
    pred=[];t=time.monotonic()
    with torch.inference_mode():
        for s in range(0,len(x),64):
            pred.append(model(x[s:s+64]).argmax(1))
            if s%1024==0:print(label,s,'/',len(x),'elapsed',round(time.monotonic()-t,1),flush=True)
    p=torch.cat(pred)
    return p,time.monotonic()-t

def main():
    torch.manual_seed(SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if (ROOT/'experiment_manifest.json').exists():raise RuntimeError('Existing completed run: choose a NEW copied run directory')
    scales,calids=calibrate()
    base=load_model();quantnames=list(scales)
    exceptions=[{'name':n,'type':type(m).__name__,'handling':'FP32 retained'} for n,m in base.named_modules() if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.ReLU,torch.nn.AdaptiveAvgPool2d))]
    exceptions.append(dict(name='residual_add',type='functional operation',handling='FP32; requantized at next learned operator input'))
    manifest=dict(run_id='int4_v1_20260907',status='running',root=str(ROOT),legacy_root=str(ROOT.parent),
        python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=4,seed=SEED,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),model_definition=str(MODEL_REPO/'pytorch_cifar_models/resnet.py'),model_definition_sha256=sha(MODEL_REPO/'pytorch_cifar_models/resnet.py'),
        dataset={s:{'path':str(DATA/s),'sha256':sha(DATA/s)} for s in ('train','test')},
        split_ids=dict(calibration_train=calids,development_test=[0,1],historical_development_test=list(range(100)),standard_test=list(range(10000)),disjoint_test=list(range(100,10000))),
        quantized_layers=quantnames,cim_mapped_layers=list(TARGETS),exceptions=exceptions,
        quantization=dict(weights='signed symmetric W4 [-7,7], per output channel absmax/7; zero point 0',activations='conv1 signed [-7,7]; all other learned operator inputs unsigned [0,15]; per tensor frozen train-calibrated scale; zero point 0',rounding='torch.round nearest ties to even',clipping='clamp to declared endpoints',bias='original FP32 after dequantization',accumulator='golden integer-valued CPU float32 convolution under checked absolute-sum bound <2**24, exported int64; independent int64 matmul validates mapped operators; ADC reconstruction float64',bn='unfolded frozen eval FP32 BN, no fusion',first_last='both W4A4; first A signed, fc A unsigned'),
        mapping=dict(label='ISAAC-inspired, not original ISAAC numerical reproduction',rows=128,columns=128,occupied_columns=[16,32,64],activation_bits=4,weight_slices=2,cell_bits=2,weight_magnitude_bits=3,top_slice_padding_bits=1,sign_branches=['positive','negative'],event_order='row_chunk, activation_bit, weight_slice, sign_branch, sample, spatial_window, occupied_column',bias_applied='once after all row chunks/bit slices/sign branches',adc_sharing='assumed one logical ADC lane and one reusable S&H bank per mapped layer; read_id defines bank contents, columns serialized; no global timing/resource schedule validated'),
        adc=dict(bits=8,raw_full_scale=384,transfer_id='linear_u8_raw384_rne_v1',code='round(clamp(raw,0,384)*255/384)',reconstruction='code*384/255',normalized_analog='raw/384',physical_voltage_mapping_status='unverified',vfs_v=None,vcm_v=None,zero_gating='disabled: every event counts as ADC conversion; preknown flags supplied from digital activation OR/static weight metadata'),
        provenance=dict(reported=['checkpoint/dataset files and original source code'],derived=['W4 slice padding','integer event reconstruction','384=128*3'],assumed=['ideal linear ADC','logical column service order','BN/residual FP32 boundaries','train max PTQ'],estimated=['legacy ADC 32 nm 1GHz ADC Plug-In 1.71371 pJ/conversion, not used to claim chip energy'],calibrated=[]),
        sources=[dict(url='https://www-old.cs.utah.edu/~rajeev/pubs/isca16.pdf',version='ISCA 2016; author-hosted PDF accessed 2026-09-07',scope='architectural context only; no numerical reproduction'),dict(url='https://github.com/mit-emze/cimloop',version='archival main README accessed 2026-09-07; no artifact commit reproduced',scope='official artifact reference, no migration')])
    savej('experiment_manifest.in_progress.json',manifest)
    dx,dy=load_data('test',[0,1]);metrics=[];outputs={};logits={};layerrows=[];qarchive={};cliprows=[]
    with torch.inference_mode():
        fp=base(dx);logits['fp_reference']=fp;fpout={};hooks=[]
        for n,m in base.named_modules():
            if n in scales:hooks.append(m.register_forward_hook(lambda m,a,o,n=n:fpout.__setitem__(n,o.clone())))
        base(dx)
        for h in hooks:h.remove()
        for mode in ('int4_digital_golden','int4_cim_exact','int4_cim_adc8_no_reuse'):
            model,state=build(scales,mode);state.collect=True
            t=time.monotonic();lg=model(dx);logits[mode]=lg;outputs[mode]=state.outputs
            print(mode,'development complete',round(time.monotonic()-t,2),flush=True)
            events_count=sum(m.stats['events'] for m in model.modules() if isinstance(m,QuantOp))
            metrics.append(dict(mode=mode,split='development_test_0_1',samples=2,correct=int((lg.argmax(1)==dy).sum()),accuracy=float((lg.argmax(1)==dy).float().mean()),adc_events=events_count,adc_conversions=events_count if mode.endswith('no_reuse') else 0))
            for n,m in model.named_modules():
                if isinstance(m,QuantOp):cliprows.append(dict(mode=mode,split='development_test_0_1',layer=n,**dict(m.stats)))
            if mode=='int4_digital_golden':
                for n,m in model.named_modules():
                    if isinstance(m,QuantOp):
                        key=n.replace('.','_');qarchive[key+'__weight_q']=m.qw.numpy().astype(np.int8);qarchive[key+'__input_q']=state.qinputs[n].numpy().astype(np.int8);qarchive[key+'__weight_scale']=m.sw.numpy();qarchive[key+'__activation_scale']=np.array(m.sa)
            if mode=='int4_cim_exact':
                assert torch.equal(lg,logits['int4_digital_golden'])
                for n in scales:assert torch.equal(state.outputs[n],outputs['int4_digital_golden'][n])
            if mode.endswith('no_reuse'):
                export_trace(state)
                for n in scales:
                    gold=outputs['int4_digital_golden'][n];exact=outputs['int4_cim_exact'][n];ad=state.outputs[n]
                    layerrows.append(dict(layer=n,samples=2,fp_to_w4a4_propagated_mse=float(((gold-fpout[n]).double()**2).mean()),w4a4_to_exact_mse=float(((exact-gold).double()**2).mean()),exact_to_adc8_propagated_mse=float(((ad-exact).double()**2).mean()),exact_to_adc8_max=float((ad-exact).abs().max()),**state.local[n]))
        metrics.insert(0,dict(mode='fp_reference',split='development_test_0_1',samples=2,correct=int((fp.argmax(1)==dy).sum()),accuracy=float((fp.argmax(1)==dy).float().mean()),adc_events=0,adc_conversions=0))
    np.savez_compressed(ROOT/'quantized_integer_sample.npz',**qarchive)
    np.savez_compressed(ROOT/'development_logits.npz',labels=dy.numpy(),**{n:v.numpy() for n,v in logits.items()})
    writecsv('layer_errors.csv',layerrows);writecsv('baseline_metrics.csv',metrics)
    # Scales have already been frozen; no test-dependent retuning occurs.
    tx,ty=load_data('test',list(range(10000)))
    allpred={}
    for mode in ('fp_reference','int4_digital_golden'):
        model=load_model() if mode=='fp_reference' else build(scales,mode)[0]
        predictions,elapsed=evaluate(model,tx,ty,mode)
        allpred[mode]=predictions.numpy()
        for split,sl in [('standard_test_all',slice(0,10000)),('disjoint_test_100_9999',slice(100,10000)),('historical_development_test_0_99',slice(0,100))]:
            n=len(ty[sl]);correct=int((predictions[sl]==ty[sl]).sum())
            metrics.append(dict(mode=mode,split=split,samples=n,correct=correct,accuracy=correct/n,adc_events=0,adc_conversions=0,elapsed_seconds_full_test=elapsed))
        for n,m in model.named_modules():
            if isinstance(m,QuantOp):cliprows.append(dict(mode=mode,split='standard_test_all',layer=n,**dict(m.stats)))
        writecsv('baseline_metrics.csv',metrics)
    np.savez_compressed(ROOT/'full_test_predictions.npz',sample_id=np.arange(10000),labels=ty.numpy(),**allpred)
    for r in cliprows:r['activation_clipping_rate']=r['activation_clipped']/r['activation_values']
    writecsv('clipping_rates.csv',cliprows)
    manifest['status']='completed';manifest['not_executed']=['full-test CIM exact/ADC8 event simulation','reuse','PEX injection','training','ADC bit sweep']
    manifest['code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')}
    savej('experiment_manifest.json',manifest)
    print('BASELINE COMPLETE',flush=True)
if __name__=='__main__':main()
