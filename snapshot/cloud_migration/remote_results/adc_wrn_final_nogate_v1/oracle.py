import numpy as np
def oracle(raw,em,de,mode,table,disturbance,threshold,lower,upper,skip):
 expect=np.zeros(raw.shape,np.float32);trace=np.full((*raw.shape,12),np.nan);s=np.zeros(16,np.int64)
 cmgrid=table[:5];dg=table[5:10];z=table[10:].reshape(5,5)
 for rt in range(raw.shape[0]):
  for bit in range(4):
   for ct in range(raw.shape[2]):
    for r in range(raw.shape[3]):
     ref=0.;prev=0;active=0
     for c in range(128):
      ix=(rt,bit,ct,r,c);s[0]+=1
      if em[rt,bit,r] or de[rt,ct,c]:s[1]+=1;continue
      rr=float(raw[ix])
      if mode==0:expect[ix]=rr;active+=1;continue
      v=rr/160.;cm=(v+ref)/2;d=v-ref;e=0.;near=False;call=active>0 and mode>=2
      valid=call and 0<=v<=1.2 and 0<=ref<=1.2
      if call:s[11]+=valid;s[12]+=not valid
      if valid:
       s[13]+=not dg[0]<=d<=dg[-1];s[14]+=not cmgrid[0]<=cm<=cmgrid[-1];s[15]+=lower<abs(d)<upper
       if disturbance:e=float(np.interp(cm,cmgrid,[np.interp(d,dg,row) for row in z]))
       near=mode==3 and abs(d)<=threshold
      gold=max(0,min(255,round(rr*255/192)));full=max(0,min(255,round(rr*255/192+e*212.5)))
      width=1<<(8-skip);q=min(max(full,prev//width*width),prev//width*width+width-1) if near else full;expect[ix]=q
      trace[ix]=[prev,v,ref,cm,d,e,v+e,near,gold,full,q,8-skip if near else 8]
      s[2]+=1;s[3]+=call;s[4]+=near;s[5]+=8-skip if near else 8;s[6]+=q!=gold;s[7]+=full!=gold;s[8]+=q!=full;s[9]+=rr>192;s[10]+=abs(q-gold)
      prev=q;ref=v+e;active+=1
 return expect,s,trace
