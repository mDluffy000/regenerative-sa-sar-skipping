import sys,json,csv,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;A1=ROOT.parent/'adc_reuse_experiment/int4_v1';sys.path.insert(0,str(A1))
from run_baseline import build,load_model,load_data,TARGETS,QuantOp,quantize,patches,dequantize
from isaac_numeric import *

class ISAACConv(torch.nn.Module):
 def __init__(self,qop,state,mode,name):super().__init__();self.q=qop;self.state=state;self.mode=mode;self.name=name
 def forward(self,x):
  q=self.q;assert float(x.min())>=0;qx=quantize(x,q.sa,0,15);pa=patches(qx,q.op)
  ex,ad,met,save=encode_and_run(pa.numpy(),q.qw.flatten(1).numpy(),save=True)
  H=(x.shape[-2]+2*q.op.padding[0]-q.op.dilation[0]*(q.op.kernel_size[0]-1)-1)//q.op.stride[0]+1
  W=(x.shape[-1]+2*q.op.padding[1]-q.op.dilation[1]*(q.op.kernel_size[1]-1)-1)//q.op.stride[1]+1
  def out(a):return dequantize(torch.from_numpy(a).transpose(1,2).reshape(len(x),q.op.out_channels,H,W),q.sa,q.sw,q.op.bias)
  actual=out(ex if self.mode=='exact' else ad);gold=out(ex)
  met.update(layer=self.name,output_mse_vs_exact_same_input=float(((actual-gold).double()**2).mean()),output_max_error=float((actual-gold).abs().max()),activation_scale=q.sa,clipped_activations=int((x>q.sa*15).sum()),activation_elements=x.numel(),sample_ids=[0,1])
  self.state['layers'].append(met);self.state['outputs'][self.name]=actual.clone();self.state['payload'][self.name]=save
  self.state['payload'][self.name]['candidate_scale']=q.sa*q.sw.numpy().astype(np.float64)
  return actual

def main():
 torch.set_num_threads(4);torch.manual_seed(20260907);torch.use_deterministic_algorithms(True)
 out=ROOT/'isaac_w4a4';out.mkdir(exist_ok=False)
 scales=json.loads((A1/'calibration.json').read_text())['scales'];x,y=load_data('test',[0,1]);alllogits={};states={};metrics=[]
 with torch.inference_mode():
  for name in ['fp_reference','int4_digital_golden','exact','adc8_no_reuse']:
   if name=='fp_reference':model=load_model();state={}
   else:
    model,_=build(scales,'int4_digital_golden');state={'layers':[],'outputs':{},'payload':{}}
    if name in ['exact','adc8_no_reuse']:
     for target in TARGETS:
      q=dict(model.named_modules())[target];parent,child=target.rsplit('.',1);setattr(dict(model.named_modules())[parent],child,ISAACConv(q,state,name,target))
   logits=model(x);alllogits[name]=logits.numpy();states[name]=state
   metrics.append(dict(mode=name,samples=2,correct=int((logits.argmax(1)==y).sum()),accuracy=float((logits.argmax(1)==y).float().mean()),requests=sum(l['requests'] for l in state.get('layers',[])),conversions=sum(l['conversions'] for l in state.get('layers',[])) if name=='adc8_no_reuse' else 0))
  np.testing.assert_array_equal(alllogits['exact'],alllogits['int4_digital_golden'])
 arrays={};index=[]
 for layer,pay in states['adc8_no_reuse']['payload'].items():
  key=layer.replace('.','_');arrays.update({key+'__'+k:v for k,v in pay.items()});index.append(dict(layer=layer,prefix=key,shape=list(pay['raw'].shape),adc_bits=8,adc_fs_raw=ADC_FS,lsb_raw=ADC_STEP,ordering='sample,window,activation_bit; per array row_chunk; physical column=weight_slice*C+output_channel',ordering_status='explicit deterministic serial schedule supplement; public model has no event scheduler',physical_voltage_status='unverified',transfer_status='linear endpoint curve supplement to guide lower-half range'))
 np.savez_compressed(out/'isaac_adc_pre_trace.npz',**arrays)
 np.savez_compressed(out/'logits.npz',labels=y.numpy(),**alllogits)
 (out/'trace_index.json').write_text(json.dumps(index,indent=2))
 (out/'numeric_results.json').write_text(json.dumps({'metrics':metrics,'layer_results':states['adc8_no_reuse']['layers'],'quantization_reference':str(A1/'experiment_manifest.json'),'full_test_fp_accuracy_reused':.6883,'full_test_w4a4_digital_accuracy_reused':.3941,'adc_full_test_executed':False,'global_logits_adc_vs_exact_mse':float(np.mean((alllogits['adc8_no_reuse'].astype(np.float64)-alllogits['exact'])**2))},indent=2))
 print(json.dumps(metrics,indent=2));print('ISAAC INTEGER EXACT AND TRACE SAVED')
if __name__=='__main__':main()
