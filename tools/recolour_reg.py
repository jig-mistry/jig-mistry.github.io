"""Poster figure for the dark page. Inside the 12 panels only the hot overlay column changes, to the violet ramp;
outside them the white sheet becomes transparent and its black lettering becomes the page's bone colour."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
src,out,chk=sys.argv[1:4]
a=np.asarray(Image.open(src).convert('RGB')).astype(np.float32)/255
h,w,_=a.shape
sat=a.max(2)-a.min(2)
cols=np.where((sat>0.25).mean(0)>0.02)[0]; x0,x1=cols.min(),cols.max()+1
v=np.clip(a.mean(2),0,1)                              # hot runs black-red-yellow-white, so the mean is monotonic
xs=np.array([0,.30,.62,.85,1]); C=np.array([[.030,.026,.060],[.150,.090,.400],[.440,.290,.930],[.800,.720,1.0],[.970,.950,1.0]])
r=np.stack([np.interp(v,xs,C[:,k]) for k in range(3)],-1)
wgt=np.clip((sat-.04)/.12,0,1)[...,None]
o=a.copy(); o[:,x0:x1]=(a*(1-wgt)+r*wgt)[:,x0:x1]
# panels: the big connected non-white blocks
lab,n=ndi.label(v<0.9)
boxes=[s for s in ndi.find_objects(lab) if (s[0].stop-s[0].start)>200 and (s[1].stop-s[1].start)>200]
print('panels',len(boxes),[(b[1].start,b[0].start,b[1].stop-b[1].start,b[0].stop-b[0].start) for b in boxes[:4]])
assert len(boxes)==12
inside=np.zeros((h,w),bool)
for b in boxes: inside[b]=True
ink=np.clip((0.97-v)/0.85,0,1)                        # lettering coverage, a little ringing removed
bone=np.array([.937,.918,.875])
ink[ndi.binary_dilation(inside,iterations=5)&~inside]=0   # the soft edge of each panel is not lettering
edge=inside&~ndi.binary_erosion(inside,iterations=3)      # nor is the rim of each panel: its black ground, no brain reaches it
print('brightest pixel in the trimmed rim, before the soft edge',float(np.percentile(v[edge],50)))
inside=inside&~edge; ink[edge]=0
rgba=np.dstack([np.where(inside[...,None],o,bone),np.where(inside,1.0,ink)])
im=Image.fromarray((rgba*255+.5).astype(np.uint8),'RGBA'); im.save(out,quality=90,alpha_quality=100,method=6)
bg=Image.new('RGBA',im.size,(8,7,15,255)); bg.alpha_composite(im); bg.convert('RGB').resize((w//2,h//2)).save(chk)
print('white-ish opaque pixels left outside panels',int(((v>.5)&~inside&(rgba[...,3]>.5)).sum()),'panels unchanged except overlay:',float(abs(rgba[...,:3]-a)[inside][:, :].mean()))
