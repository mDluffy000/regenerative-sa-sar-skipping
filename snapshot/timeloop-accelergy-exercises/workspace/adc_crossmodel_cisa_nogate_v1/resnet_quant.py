"""Strict all-operator W4A4 QAT, compatible with the existing integer contract.

LSQ-style normalized scale gradients, per-output-channel weight scales and
per-tensor activation scales. Original [-7,7] symmetric signed convention is
preserved rather than switching to [-8,7]. No SA or finite ADC is simulated here.
"""
from pathlib import Path
import json,math,sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent;W=ROOT.parent
MODEL_REPO=Path('/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
CHECKPOINT=Path('/root/.cache/torch/hub/checkpoints/cifar100_resnet20-23dac2f1.pt')
CALIBRATION=W/'adc_reuse_experiment/int4_v1/calibration.json'
TARGETS=['layer1.0.conv1','layer2.1.conv2','layer3.2.conv2']

def base_model():
    sys.path.insert(0,str(MODEL_REPO))
    from pytorch_cifar_models import cifar100_resnet20
    model=cifar100_resnet20(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    return model

def scale_gradient(x,factor):return x.detach()+(x-x.detach())*factor
def round_ste(x):return x+(x.round()-x).detach()

class QuantOp(nn.Module):
    def __init__(self,name,op,activation_scale):
        super().__init__();self.name=name;self.isconv=isinstance(op,nn.Conv2d)
        self.weight=nn.Parameter(op.weight.detach().clone())
        self.bias=nn.Parameter(op.bias.detach().clone()) if op.bias is not None else None
        axes=tuple(range(1,op.weight.ndim))
        self.weight_scale=nn.Parameter(op.weight.detach().abs().amax(axes,keepdim=True).clamp_min(1e-12)/7)
        # Float64 scalar preserves old Python-float dequantization at initialization.
        self.activation_scale=nn.Parameter(torch.tensor(activation_scale,dtype=torch.float64))
        self.low=-7 if name=='conv1' else 0;self.high=7 if name=='conv1' else 15
        if self.isconv:
            self.stride=op.stride;self.padding=op.padding;self.dilation=op.dilation;self.groups=op.groups
        self.exact_eval=False

    def scales(self,x):
        sa=scale_gradient(self.activation_scale.clamp_min(1e-8),1/math.sqrt(x.numel()*self.high))
        sw=scale_gradient(self.weight_scale.clamp_min(1e-8),1/math.sqrt(self.weight[0].numel()*7))
        return sa,sw

    def integers(self,x):
        sa,sw=self.scales(x)
        qx=round_ste(torch.clamp(x/sa,self.low,self.high))
        qw=round_ste(torch.clamp(self.weight/sw,-7,7))
        return qx,qw,sa,sw

    def forward(self,x):
        qx,qw,sa,sw=self.integers(x)
        if self.exact_eval:
            assert not self.training
            # Independent integer-valued path. Conv float accumulations exact under bound.
            qx=torch.round(x/self.activation_scale.clamp_min(1e-8)).clamp(self.low,self.high).to(torch.int64)
            qw=torch.round(self.weight/self.weight_scale.clamp_min(1e-8)).clamp(-7,7).to(torch.int64)
            assert self.weight[0].numel()*self.high*7<2**24
            qx=qx.float();qw=qw.float()
        if self.isconv:
            acc=F.conv2d(qx,qw,None,self.stride,self.padding,self.dilation,self.groups)
            out=(acc.double()*sa*sw.flatten().double()[None,:,None,None]).float()
            if self.bias is not None:out=out+self.bias[None,:,None,None]
        else:
            acc=F.linear(qx,qw)
            out=(acc.double()*sa*sw.flatten().double()[None,:]).float()
            if self.bias is not None:out=out+self.bias
        return out

def build():
    scales=json.loads(CALIBRATION.read_text())['scales'];model=base_model()
    for name,op in list(model.named_modules()):
        if isinstance(op,(nn.Conv2d,nn.Linear)):
            parent,_,child=name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model,child,QuantOp(name,op,scales[name]))
    assert sum(isinstance(m,QuantOp) for m in model.modules())==22
    return model

def set_exact(model,enabled):
    for m in model.modules():
        if isinstance(m,QuantOp):m.exact_eval=enabled

def export_quantizers(model):
    return {name:dict(weight_bits=4,activation_bits=4,weight_range=[-7,7],activation_range=[m.low,m.high],
        activation_scale=float(m.activation_scale.detach()),weight_scale=m.weight_scale.detach().flatten().tolist())
        for name,m in model.named_modules() if isinstance(m,QuantOp)}
