"""Dependency-free DeiT-Tiny topology; official state keys, independent torchvision check.

Uses the existing PyTorch runtime, not an installation or replacement of timm.
"""
from pathlib import Path
import math
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
R=Path(__file__).resolve().parent
def configure_runtime():
 # On this emulated CPU, default fused paths differ between layouts. Freeze
 # standard math kernels for reference verification and all subsequent runs.
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.mkldnn.enabled=False
 torch.backends.mha.set_fastpath_enabled(False)
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)

class PatchEmbed(nn.Module):
 def __init__(self):
  super().__init__();self.proj=nn.Conv2d(3,192,16,16)
 def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)
class Attention(nn.Module):
 def __init__(self):
  super().__init__();self.qkv=nn.Linear(192,576);self.proj=nn.Linear(192,192)
 def forward(self,x):
  b,n,c=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,3,64).permute(2,0,3,1,4).unbind(0)
  z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(b,n,c)
  return self.proj(z)
class Mlp(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(192,768);self.fc2=nn.Linear(768,192)
 def forward(self,x):return self.fc2(F.gelu(self.fc1(x)))
class Block(nn.Module):
 def __init__(self):
  super().__init__();self.norm1=nn.LayerNorm(192,eps=1e-6);self.attn=Attention();self.norm2=nn.LayerNorm(192,eps=1e-6);self.mlp=Mlp()
 def forward(self,x):
  x=x+self.attn(self.norm1(x));return x+self.mlp(self.norm2(x))
class DeiTTiny(nn.Module):
 def __init__(self,classes=1000):
  super().__init__();self.patch_embed=PatchEmbed();self.cls_token=nn.Parameter(torch.zeros(1,1,192));self.pos_embed=nn.Parameter(torch.zeros(1,197,192))
  self.blocks=nn.ModuleList([Block() for _ in range(12)]);self.norm=nn.LayerNorm(192,eps=1e-6);self.head=nn.Linear(192,classes)
 def features(self,x):
  x=self.patch_embed(x);x=torch.cat([self.cls_token.expand(len(x),-1,-1),x],1)+self.pos_embed
  for blk in self.blocks:x=blk(x)
  return self.norm(x)[:,0]
 def forward(self,x):return self.head(self.features(x))
def pretrained():
 m=DeiTTiny();state=torch.load(R/'input/deit_tiny_patch16_224-a1311bcf.pth',weights_only=True,map_location='cpu')['model'];m.load_state_dict(state,strict=True);return m
def reference_torchvision(state):
 from torchvision.models.vision_transformer import VisionTransformer
 m=VisionTransformer(image_size=224,patch_size=16,num_layers=12,num_heads=3,hidden_dim=192,mlp_dim=768,num_classes=1000,norm_layer=partial(nn.LayerNorm,eps=1e-6))
 target={'class_token':state['cls_token'],'encoder.pos_embedding':state['pos_embed']}
 mapping={'conv_proj':'patch_embed.proj','encoder.ln':'norm','heads.head':'head'}
 for i in range(12):
  t=f'encoder.layers.encoder_layer_{i}';s=f'blocks.{i}'
  mapping.update({t+'.ln_1':s+'.norm1',t+'.ln_2':s+'.norm2',t+'.self_attention.out_proj':s+'.attn.proj',t+'.mlp.0':s+'.mlp.fc1',t+'.mlp.3':s+'.mlp.fc2'})
  for kind in ['weight','bias']:target[t+'.self_attention.in_proj_'+kind]=state[s+'.attn.qkv.'+kind]
 for t,s in mapping.items():
  for kind in ['weight','bias']:target[t+'.'+kind]=state[s+'.'+kind]
 assert len(target)==len(state);m.load_state_dict(target,strict=True);return m

def scale_gradient(x,factor):return x.detach()+(x-x.detach())*factor
def round_ste(x):return x+(x.round()-x).detach()
class QuantOp(nn.Module):
 """All fixed-weight projections W4A4, signed activations [-7,7]."""
 def __init__(self,name,op,activation_scale):
  super().__init__();self.name=name;self.isconv=isinstance(op,nn.Conv2d);self.hw=None
  self.weight=nn.Parameter(op.weight.detach().clone());self.bias=nn.Parameter(op.bias.detach().clone()) if op.bias is not None else None
  dims=tuple(range(1,op.weight.ndim))
  self.weight_scale=nn.Parameter(op.weight.detach().abs().amax(dims,keepdim=True).clamp_min(1e-8)/7)
  self.activation_scale=nn.Parameter(torch.tensor(float(activation_scale),device=op.weight.device))
  if self.isconv:self.stride=op.stride;self.padding=op.padding;self.dilation=op.dilation;self.groups=op.groups
 def integers(self,x):
  sa=scale_gradient(self.activation_scale.clamp_min(1e-8),1/math.sqrt(x.numel()*7))
  sw=scale_gradient(self.weight_scale.clamp_min(1e-8),1/math.sqrt(self.weight[0].numel()*7))
  return round_ste((x/sa).clamp(-7,7)),round_ste((self.weight/sw).clamp(-7,7)),sa,sw
 def forward(self,x):
  qx,qw,sa,sw=self.integers(x)
  if self.isconv:acc=F.conv2d(qx,qw,None,self.stride,self.padding,self.dilation,self.groups)
  else:acc=F.linear(qx,qw)
  if self.hw is not None and self.hw.mode!='digital':
   physical=self.hw.accumulator(self,qx.detach(),qw.detach());acc=physical.to(acc.dtype)+(acc-acc.detach())
  shape=(1,-1,1,1) if self.isconv else tuple([1]*(acc.ndim-1)+[-1])
  z=acc*sa*sw.flatten().view(shape)
  if self.bias is not None:z=z+self.bias.view(shape)
  return z
def quantize(model,scales):
 for name,op in list(model.named_modules()):
  if isinstance(op,(nn.Conv2d,nn.Linear)):
   parent,_,child=name.rpartition('.');setattr(model.get_submodule(parent) if parent else model,child,QuantOp(name,op,scales[name]))
 assert sum(isinstance(x,QuantOp) for x in model.modules())==50
 return model
