"""Explicit W4A4 integer semantics and ISAAC-inspired 128-row mapping.
No reuse. CPU float32 conv is an exact integer simulation under the checked
absolute reduction bound < 2**24; independent golden validation uses int64.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

ROWS = 128
ADC_FS = 384.0

def quantize(x, scale, low, high):
    return torch.round(x / scale).clamp(low, high).to(torch.int64)

def weight_quantize(w):
    axes = tuple(range(1, w.ndim))
    scale = w.abs().amax(dim=axes, keepdim=True).clamp_min(1e-12) / 7
    return quantize(w, scale, -7, 7), scale.reshape(-1)

def patches(qx, conv):
    # Inputs are at most 15: float conversion/unfold is exact, padding is q=0.
    return F.unfold(qx.float(), conv.kernel_size, dilation=conv.dilation,
                    padding=conv.padding, stride=conv.stride).transpose(1, 2).to(torch.int64)

def integer_golden(qx, qw, conv=None):
    reduction = qw[0].numel()
    assert reduction * int(qx.abs().max()) * int(qw.abs().max()) < 2**24
    if conv is None:
        result = F.linear(qx.float(), qw.float())
    else:
        assert conv.groups == 1
        result = F.conv2d(qx.float(), qw.float(), None, conv.stride, conv.padding, conv.dilation)
    assert torch.equal(result, result.round())
    return result.to(torch.int64)

def events(x, w):
    """x=[N,P,R] A4 unsigned; w=[C,R] signed W4. Emit all physical events."""
    assert x.dtype == w.dtype == torch.int64
    assert int(x.min()) >= 0 and int(x.max()) <= 15
    assert int(w.min()) >= -7 and int(w.max()) <= 7
    assert x.shape[-1] == w.shape[-1] and w.shape[0] <= 128
    for start in range(0, x.shape[-1], ROWS):
        xc, wc = x[..., start:start+ROWS], w[:, start:start+ROWS]
        for bit in range(4):
            xb = (xc >> bit) & 1
            az = xb.sum(-1) == 0
            for sl in range(2):
                mag = (wc.abs() >> (2*sl)) & 3
                if sl == 1:
                    assert int(mag.max()) <= 1  # W4 magnitude uses only three bits.
                for branch in range(2):
                    digits = torch.where(wc >= 0 if branch == 0 else wc < 0, mag, 0)
                    wz = digits.sum(-1) == 0
                    evidence = az[...,None].to(torch.uint8) | (wz[None,None,:].to(torch.uint8) << 1)
                    raw = xb @ digits.T
                    assert bool((raw[evidence != 0] == 0).all())
                    code = torch.round(raw.float().clamp(0,ADC_FS) * (255/ADC_FS)).to(torch.uint8)
                    yield dict(row_chunk=start//ROWS, bit=bit, slice=sl, branch=branch,
                               raw=raw, code=code, evidence=evidence,
                               shift=(1 if branch==0 else -1)*(1 << (bit+2*sl)))

def cim_reconstruct(x, w, collect=False):
    shape = (*x.shape[:-1], w.shape[0])
    exact = torch.zeros(shape, dtype=torch.int64)
    adc = torch.zeros(shape, dtype=torch.float64)
    saved=[]
    count=zeros=clipped=endpoint=0
    for e in events(x,w):
        exact += e['raw'] * e['shift']
        adc += e['code'].double() * (ADC_FS/255) * e['shift']
        count += e['raw'].numel()
        zeros += int((e['evidence'] != 0).sum())
        clipped += int(((e['raw']<0)|(e['raw']>ADC_FS)).sum())
        endpoint += int((e['code']==255).sum())
        if collect: saved.append(e)
    # Independent direct integer matrix multiply, checked every CIM invocation.
    direct = x @ w.T
    assert torch.equal(exact, direct), 'CIM mapping integer mismatch'
    return exact, adc, dict(events=count,zero_preknown=zeros,adc_clipped=clipped,adc_upper_code=endpoint),saved

def dequantize(acc, a_scale, w_scale, bias=None):
    out = (acc.double() * float(a_scale) * w_scale.double()[None,:,None,None]).float()
    if bias is not None: out += bias[None,:,None,None]
    return out
