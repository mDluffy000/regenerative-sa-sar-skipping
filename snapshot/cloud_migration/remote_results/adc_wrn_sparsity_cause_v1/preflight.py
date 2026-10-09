from run import *
def preflight():
 g=np.random.default_rng(914);raw=g.integers(0,50,(2,4,2,4,128)).astype(np.float32);raw[:,2]=0
 raw[...,::17]=g.integers(180,385,raw[...,::17].shape);raw[...,0:8]=[0,0,3,4,8,9,192,384]
 em=np.zeros((2,4,4),np.uint8);de=np.zeros((2,2,128),np.uint8);de[:,1,96:]=1
 ids=torch.tensor([7,9],device='cuda');args=[torch.from_numpy(v).cuda() for v in [raw,em,de]];checks=[]
 valid=2*4*4*(128+96);padding=raw.size-valid
 for target in TARGETS:
  for skip in SKIPS:
   for mode,dist in [(0,False),(1,False),(2,True),(3,False),(3,True)]:
    h=Hardware(target=target,skip=skip,disturbance=dist)
    out,s,tr=hwmod.EXT.convert(*args,ids,2,3,0,mode,skip,h.theta,0.,True,h.table,dist,h.lower,h.upper)
    aa,ss,tt=oracle(raw,em,de,mode,h.table.cpu().numpy(),dist,h.theta,h.lower,h.upper,skip)
    assert np.array_equal(out.cpu().numpy(),aa) and np.array_equal(s.cpu().numpy(),ss),(target,skip,mode,'codes/counts')
    np.testing.assert_allclose(tr.cpu().numpy(),tt,rtol=0,atol=1e-11,equal_nan=True)
    assert ss[1]==padding
    if mode:assert ss[2]==valid and ss[5]==8*valid-skip*ss[4]
    checks.append(dict(target=target,skip=skip,mode=mode,disturbance=dist,events=raw.size))
 t.save('preflight.json',dict(status='PASS',checks=checks,events_checked=sum(v['events'] for v in checks),padding_only_exclusion=True,zero_samples_still_converted=True))

