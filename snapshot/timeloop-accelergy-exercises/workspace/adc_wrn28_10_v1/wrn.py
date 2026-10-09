"""WRN-28-10, no dropout, preactivation shortcut as author's PyTorch model.

Topology source: szagoruyko/wide-residual-networks/pytorch/resnet.py.
Random initialization, no CIFAR-trained external checkpoint.
"""
import sys,math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
BASE=Path('/root/autodl-tmp/adc_deit_cloud_v1')
sys.path.insert(0,str(BASE))
from model import QuantOp

class Block(nn.Module):
 def __init__(self,ci,co,stride):
  super().__init__();self.bn1=nn.BatchNorm2d(ci);self.conv1=nn.Conv2d(ci,co,3,stride,1,bias=False)
  self.bn2=nn.BatchNorm2d(co);self.conv2=nn.Conv2d(co,co,3,1,1,bias=False)
  self.shortcut=nn.Conv2d(ci,co,1,stride,bias=False) if ci!=co else None
 def forward(self,x):
  y=F.relu(self.bn1(x));z=self.conv2(F.relu(self.bn2(self.conv1(y))))
  return z+(self.shortcut(y) if self.shortcut is not None else x)

class WRN(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Conv2d(3,16,3,1,1,bias=False);self.groups=nn.ModuleList();ci=16
  for g,co in enumerate([160,320,640]):
   self.groups.append(nn.Sequential(*[Block(ci if j==0 else co,co,2 if g>0 and j==0 else 1) for j in range(4)]));ci=co
  self.bn=nn.BatchNorm2d(640);self.fc=nn.Linear(640,100)
  for m in self.modules():
   if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
   elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
   elif isinstance(m,nn.Linear):nn.init.zeros_(m.bias)
 def forward(self,x):
  x=self.conv(x)
  for g in self.groups:x=g(x)
  return self.fc(F.avg_pool2d(F.relu(self.bn(x)),8).flatten(1))

def quantize(m,scales):
 for name,op in list(m.named_modules()):
  if isinstance(op,(nn.Conv2d,nn.Linear)):
   parent,_,child=name.rpartition('.');setattr(m.get_submodule(parent) if parent else m,child,QuantOp(name,op,scales[name]))
 assert sum(isinstance(op,QuantOp) for op in m.modules())==29
 return m

TARGETS=['groups.0.3.conv2','groups.1.3.conv2','groups.2.3.conv2']
def attach(m,hw):
 ops=[(n,op) for n,op in m.named_modules() if isinstance(op,QuantOp)]
 hw.names={n:i for i,(n,_) in enumerate(ops)}
 for n,op in ops:op.hw=hw if n in TARGETS else None
 assert set(TARGETS)<=hw.names.keys()

def runtime():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)

def prep(x,augment=False):
 # Input stays native CIFAR 32x32; all arithmetic here is on GPU.
 x=x.cuda(non_blocking=True).float()/255
 if augment:
  x=F.pad(x,(4,4,4,4));o=torch.randint(9,(len(x),2),device=x.device)
  rr=torch.arange(32,device=x.device)[None,:,None]+o[:,0,None,None]
  cc=torch.arange(32,device=x.device)[None,None,:]+o[:,1,None,None]
  x=x.permute(0,2,3,1)[torch.arange(len(x),device=x.device)[:,None,None],rr,cc].permute(0,3,1,2)
  flip=torch.rand(len(x),device=x.device)<.5;x=torch.where(flip[:,None,None,None],x.flip(-1),x)
 mean=x.new_tensor([.5071,.4867,.4408])[None,:,None,None];std=x.new_tensor([.2675,.2565,.2761])[None,:,None,None]
 return ((x-mean)/std).contiguous()
