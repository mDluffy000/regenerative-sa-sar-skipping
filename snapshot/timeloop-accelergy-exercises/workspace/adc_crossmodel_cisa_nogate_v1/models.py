"""Original model definitions and quantizers, with an explicit hardware forward."""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
import resnet_quant as rq
import deit_model as dm
import deit_data as dd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'input'))
import resnet_source
TARGETS = {
    'resnet': ['layer1.0.conv1', 'layer2.1.conv2', 'layer3.2.conv2'],
    'deit': ['blocks.0.mlp.fc2', 'blocks.5.mlp.fc2', 'blocks.11.mlp.fc2'],
}

class ResHardwareOp(rq.QuantOp):
    def forward(self, x):
        qx, qw, sa, sw = self.integers(x)
        exact = F.conv2d(qx, qw, None, self.stride, self.padding, self.dilation, self.groups)
        if self.hw is None or self.hw.mode == 'digital':
            acc = exact.double()
        else:
            physical = self.hw.accumulator(self, qx.detach(), qw.detach())
            acc = physical.double() + (exact.double() - exact.double().detach())
        out = (acc * sa * sw.flatten().double()[None, :, None, None]).float()
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out

def build(kind):
    checkpoint = R/'input'/('resnet_W4A4.pt' if kind == 'resnet' else 'deit_W4A4_best.pt')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    if kind == 'resnet':
        m = resnet_source.cifar100_resnet20(pretrained=False)
        for name, op in list(m.named_modules()):
            if isinstance(op, (nn.Conv2d, nn.Linear)):
                parent, _, child = name.rpartition('.')
                new = rq.QuantOp(name, op, float(state[name+'.activation_scale']))
                if name in TARGETS[kind]:
                    new.__class__ = ResHardwareOp
                    new.hw = None
                setattr(m.get_submodule(parent) if parent else m, child, new)
    else:
        m = dm.DeiTTiny(classes=100)
        scales = {n[:-17]: float(v) for n,v in state.items() if n.endswith('.activation_scale')}
        m = dm.quantize(m, scales)
    m.load_state_dict(state, strict=True)
    return m.cuda().eval()

def attach(model, hw, kind):
    hw.names = {n:i for i,n in enumerate(TARGETS[kind])}
    for name in TARGETS[kind]:
        model.get_submodule(name).hw = hw

def detach(model, kind):
    for name in TARGETS[kind]:
        model.get_submodule(name).hw = None

def prep(x, kind, augment=False):
    # Preserve each old model's normalization, resize and reflection-pad augmentation.
    if kind == 'deit':
        return dd.preprocess(x, augment).cuda().contiguous()
    x = x.float()/255
    if augment:
        padded = F.pad(x, (4,4,4,4), mode='reflect')
        offset = torch.randint(9, (len(x),2))
        x = torch.stack([padded[i,:,int(y):int(y)+32,int(z):int(z)+32]
                         for i,(y,z) in enumerate(offset)])
        flip = torch.rand(len(x)) < .5
        x[flip] = x[flip].flip(-1)
    mean = x.new_tensor([.5071,.4867,.4408])[None,:,None,None]
    std = x.new_tensor([.2675,.2565,.2761])[None,:,None,None]
    return ((x-mean)/std).cuda().contiguous()

def runtime():
    dm.configure_runtime()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
