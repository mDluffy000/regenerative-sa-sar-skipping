from pathlib import Path
import hashlib,json,pickle
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parent;W=R.parent
def read_data(split):
 with (W/'bf16_fp_cim_reproduction/data/cifar-100-python'/split).open('rb') as f:d=pickle.load(f,encoding='bytes')
 return torch.from_numpy(d[b'data'].copy()).reshape(-1,3,32,32),torch.tensor(d[b'fine_labels'])
def preprocess(x,augment=False):
 x=x.float()/255
 if augment:
  padded=F.pad(x,(4,4,4,4),mode='reflect');offset=torch.randint(9,(len(x),2))
  x=torch.stack([padded[i,:,int(y):int(y)+32,int(z):int(z)+32] for i,(y,z) in enumerate(offset)])
  flip=torch.rand(len(x))<.5;x[flip]=x[flip].flip(-1)
 x=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
 return (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
def create_split():
 p=R/'split.json'
 if p.exists():return json.loads(p.read_text())
 x,y=read_data('train');g=np.random.default_rng(20260910)
 # Prior audit identified ten exact train/test duplicates. Exclude their TRAIN IDs.
 excluded={24083,32205,28538,30697,28189,21619,29363,25390,37654,15597}
 seen={};duplicates=[]
 for i,a in enumerate(x.numpy()):
  h=hashlib.sha256(a.tobytes()).hexdigest()
  if h in seen:excluded.add(i);duplicates.append([seen[h],i])
  else:seen[h]=i
 val=[]
 for c in range(100):
  eligible=[i for i in np.flatnonzero(y.numpy()==c) if i not in excluded]
  val.extend(g.choice(eligible,50,replace=False).tolist())
 train=sorted(set(range(50000))-excluded-set(val))
 out=dict(train=train,validation=sorted(val),excluded_train_ids=sorted(excluded),train_content_duplicates=duplicates,
  source='official train only; no test labels used for split, calibration, selection or training',seed=20260910)
 assert not set(train)&set(val)
 p.write_text(json.dumps(out,indent=2)+'\n');return out
def subset(y,eligible,n,seed):
 g=np.random.default_rng(seed);eligible=np.asarray(eligible)
 return np.sort(np.concatenate([g.choice(eligible[y.numpy()[eligible]==c],n,replace=False) for c in range(100)]))
