"""Poster figure: only the hot overlay column changes, to the page's violet ramp. Everything else stays as published."""
import sys, numpy as np
from PIL import Image
src,out,chk=sys.argv[1:4]
a=np.asarray(Image.open(src).convert('RGB')).astype(np.float32)/255
h,w,_=a.shape
sat=a.max(2)-a.min(2)
cols=np.where((sat>0.25).mean(0)>0.02)[0]; x0,x1=cols.min(),cols.max()+1
print('overlay columns',x0,x1,'of',w,flush=True)
v=np.clip((a[...,0]+a[...,1]+a[...,2])/3,0,1)        # hot runs black-red-yellow-white, so the mean is monotonic
xs=np.array([0,.30,.62,.85,1]); C=np.array([[.030,.026,.060],[.150,.090,.400],[.440,.290,.930],[.800,.720,1.0],[.970,.950,1.0]])
r=np.stack([np.interp(v,xs,C[:,k]) for k in range(3)],-1)
wgt=np.clip((sat-.04)/.12,0,1)[...,None]
o=a.copy(); o[:,x0:x1]=(a*(1-wgt)+r*wgt)[:,x0:x1]
im=Image.fromarray((o*255+.5).astype(np.uint8)); im.save(out,quality=88,method=6)
im.resize((w//2,h//2)).save(chk)
