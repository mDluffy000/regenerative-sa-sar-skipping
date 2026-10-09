"""Fixed five-epoch follow-up after train-only recalibration; no test access."""
from pathlib import Path
import sys,json,copy,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
import run as t
from models import build,runtime,detach
from hardware import Hardware
from deit_data import read_data
runtime();torch.manual_seed(t.SEED);t.compile_kernel()
O=R/'diagnostics/bn_refresh_finetune';O.mkdir(exist_ok=False)
t.OUT=O;t.EPOCHS=5;t.STATE=dict(model='ResNet20_BN_refresh_diagnostic',status='running',stages={},test={})
protocol=dict(epochs=5,train_images=2000,validation_images=500,test_used=False,branches=['native_ADC','SA20_skip4','SA40_skip6'],initialization='same original W4A4 parent plus per-route one-pass BN recalibration, see bn_refresh/PROTOCOL.json',optimizer='same AdamW1e-5 to1e-6 and CE+0.5*T^2*KL, T2',selection='validation accuracy then CE, epoch0 eligible; all three routes reported; no test results',note='Exploratory protocol repair, separate from frozen primary matrix; not additional evidence from independent test data')
t.save('PROTOCOL.json',protocol)
x,y=read_data('train');s=json.loads((R/'resnet/split.json').read_text());ids=np.array(s['train']);valid=np.array(s['validation'])
teacher=build('resnet').eval()
for p in teacher.parameters():p.requires_grad_(False)
for name,target,skip in [('native_ADC',None,4),('SA20_skip4',20,4),('SA40_skip6',40,6)]:
 m=build('resnet');m.load_state_dict(torch.load(R/'diagnostics/bn_refresh'/(name+'.pt'),map_location='cpu',weights_only=True)['model'])
 hw=Hardware('adc',disturbance=False) if target is None else Hardware(target=target,skip=skip)
 t.train('resnet',name,m,teacher,hw,x,y,ids,valid)
 m.load_state_dict(torch.load(O/(name+'_best.pt'),map_location='cpu',weights_only=True)['model'])
 v=t.evaluate('resnet',m,x,y,valid,hw,name+'_selected_validation');t.save(name+'_selected_validation.json',v)
 del m,hw;torch.cuda.empty_cache()
t.STATE['status']='complete';t.report();t.status('complete')
