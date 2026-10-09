"""ISAAC CiMLoop offset encoding, native 1b input / 2b cell / 8b ADC.
Event order is an explicit legal schedule, not an unavailable public cycle trace.
Linear ADC half-range transfer is a disclosed extension of guide prose.
"""
import numpy as np
ROWS=128;COLS=128;ADC_BITS=8;ADC_FS=192.;ADC_STEP=ADC_FS/255.;ADC_PERIOD_NS=1/1.28

def encode_and_run(x,w,a_offset=0,w_offset=7,save=False):
    x=np.asarray(x,np.int64);w=np.asarray(w,np.int64)
    assert x.ndim==3 and w.ndim==2 and x.shape[-1]==w.shape[-1]
    assert np.min(x+a_offset)>=0 and np.max(x+a_offset)<=15
    assert np.min(w+w_offset)>=0 and np.max(w+w_offset)<=15
    N,P,R=x.shape;M=w.shape[0];assert 2*M<=128
    exact=np.zeros((N,P,M),np.int64);adc=np.zeros_like(exact,np.float64)
    records=[];raws=[];codes=[];corrections=[];metrics={'requests':0,'conversions':0,'clipped':0,'integer_mismatch':0,'preknown_zero':0}
    # Loop order in storage is read n,p,abit; arrays independent; columns Y then M.
    for rc,start in enumerate(range(0,R,128)):
        a=x[...,start:start+128]+a_offset;b=w[:,start:start+128]+w_offset
        correction=-w_offset*a.sum(-1)[...,None]-a_offset*b.sum(-1)[None,None,:]+a.shape[-1]*a_offset*w_offset
        ex=np.zeros_like(exact);ad=np.zeros_like(adc)
        for bit in range(4):
            xb=(a>>bit)&1
            for sl in range(2):
                digit=(b>>(2*sl))&3;raw=xb@digit.T
                code=np.rint(np.clip(raw,0,ADC_FS)*255/ADC_FS).astype(np.uint8)
                shift=1<<(bit+2*sl)
                ex+=raw*shift;ad+=code.astype(np.float64)*ADC_STEP*shift
                z=(xb.sum(-1)==0)[...,None]|(digit.sum(-1)==0)[None,None,:]
                metrics['requests']+=raw.size;metrics['conversions']+=raw.size;metrics['clipped']+=int((raw>ADC_FS).sum());metrics['preknown_zero']+=int(z.sum())
                if save:records.append((rc,bit,sl,shift));raws.append(raw.astype(np.int16));codes.append(code)
        exact+=ex+correction;adc+=ad+correction;corrections.append(correction)
    direct=x@w.T;metrics['integer_mismatch']=int((exact!=direct).sum());assert np.array_equal(exact,direct)
    payload={}
    if save:
        payload={'raw':np.stack(raws),'code':np.stack(codes),'meta':np.array(records,np.int64),'offset_corrections':np.stack(corrections),'input_q':x.astype(np.int8),'weight_q':w.astype(np.int8),'exact':exact,'adc':adc,'activation_offset':np.array(a_offset),'weight_offset':np.array(w_offset)}
    metrics['adc_local_integer_units_mse']=float(np.mean((adc-exact)**2))
    return exact,adc,metrics,payload
