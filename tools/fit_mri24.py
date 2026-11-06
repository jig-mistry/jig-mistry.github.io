"""Fit pipeline space (sub-005_final_stack) -> web grid of dharani_24gw, and report it. Low-res, small memory."""
import sys, gzip, json, numpy as np, nibabel as nib
from scipy import ndimage as ndi
from scipy.optimize import minimize
R='/mnt/28466BE9466BB5EA/mcmadras_ppline/madras_run/sub-005_final/'
A=sys.argv[1]; OUT=sys.argv[2]
meta=[b for b in json.load(open(A+'/volumes/brains.json'))['brains'] if b['id']=='dharani_24gw'][0]
u,v,w=meta['grid']; sp=meta['voxel_mm']
web=np.frombuffer(gzip.open(A+'/volumes/dharani_24gw.bin.gz').read(),np.uint8).reshape(w,v,u,2)
dens=web[...,0].astype(np.float32)/255; 
print('web',dens.shape,flush=True)
def load(p,step):
    im=nib.load(p); d=im.dataobj
    a=np.asarray(d[::step[0],::step[1],::step[2]] ,dtype=np.float32)
    return a,im.affine,step
s,aff,st=load(R+'sub-005_final_stack.mnc',(1,1,1)) if False else load(R+'sub-005_final_stack.mnc',(4,2,2))
print('stack',s.shape,s.min(),s.max(),'\n',aff,flush=True)
aff2=aff.copy(); aff2[:3,:3]=aff[:3,:3]*np.array(st)
inv=np.linalg.inv(aff2)
bg=np.percentile(s,50); print('median',bg, 'p5,95',np.percentile(s,[5,95]))
s=ndi.gaussian_filter(s,1)
# coarse web grid
k=4
D=dens[::k,::k,::k]
ww,vv,uu=np.meshgrid(*[ (np.arange(n)[::k]-(n-1)/2)*sp for n in (w,v,u)],indexing='ij')
q=np.stack([uu.ravel(),vv.ravel(),ww.ravel()])
Dm=D.ravel()
in_=Dm>0.004
def sample(T,c):
    p=T@q+c[:,None]
    ijk=inv[:3,:3]@p+inv[:3,3:4]
    return ndi.map_coordinates(s,ijk,order=1,mode='constant',cval=float(bg))
def ncc(a,b):
    a=a-a.mean();b=b-b.mean();return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))
def rotm(ax,t):
    c,n=np.cos(t),np.sin(t)
    return {0:np.array([[1,0,0],[0,c,-n],[0,n,c]]),1:np.array([[c,0,n],[0,1,0],[-n,0,c]]),2:np.array([[c,-n,0],[n,c,0],[0,0,1]])}[ax]
# world centre of the tissue
ctr=(aff2@np.r_[(np.array(s.shape)-1)/2,1])[:3]
# stack axes order from nibabel: i=z,j=y,k=x slowest first? use affine; web u along stack section axis
best=[]
secAxis=np.argmax(np.abs(aff[:3,0]))   # world axis of the section stacking direction
print('section world axis',secAxis)
others=[a for a in range(3) if a!=secAxis]
for su in (1,-1):
  for sw in (1,-1):
    for ang in range(0,360,10):
        t=np.deg2rad(ang);c,n=np.cos(t),np.sin(t)
        T=np.zeros((3,3)); T[secAxis,0]=su
        T[others[0],1]=c; T[others[0],2]=-n*sw
        T[others[1],1]=n; T[others[1],2]=c*sw
        val=sample(T,ctr)
        best.append((abs(ncc(val[in_],Dm[in_]))+abs(ncc(val,Dm)),ncc(val,Dm),su,sw,ang,T))
best.sort(key=lambda b:-b[0])
for b in best[:6]: print(round(b[0],4),round(b[1],4),b[2:5],flush=True)
def params(x,T0):
    return T0@rotm(0,x[0])@rotm(1,x[1])@rotm(2,x[2])*x[6], ctr+x[3:6]
res=None
for b in best[:3]:
    f=lambda x:-abs(ncc(sample(*params(x,b[5])),Dm))
    r=minimize(f,[0,0,0,0,0,0,1],method='Powell',options=dict(maxiter=8,xtol=1e-3,ftol=1e-4))
    print('refined',-r.fun,r.x.round(3),flush=True)
    if res is None or r.fun<res[0].fun: res=(r,b)
T,c=params(res[0].x,res[1][5])
sign=float(np.sign(ncc(sample(T,c),Dm)))
print('final ncc',ncc(sample(T,c),Dm),'det',np.linalg.det(T))
np.savez(OUT,T=T,c=c,sign=sign)
