"""24-week T1 on the web grid straight from the NATIVE scan: one rigid placement, no scale, shear or warp, one resampling."""
import sys, gzip, json, os, numpy as np, nibabel as nib
from scipy import ndimage as ndi
from scipy.optimize import minimize
R='/mnt/28466BE9466BB5EA/mcmadras_ppline/madras_run/sub-005_final/'
A,FIT,OUT,PNG=sys.argv[1:5]
f=np.load(FIT); T,c=f['T'],f['c']
Tr=T/np.cbrt(abs(np.linalg.det(T)))                  # drop the 0.7% scale of the fit: rotation + the display mirror only
t=open(R+'mri/mri_native_to_histo_step3.xfm').read().split('Linear_Transform =')[1].replace(';','').split()
M=np.vstack([np.array(t[:12],float).reshape(3,4),[0,0,0,1]]); Mi=np.linalg.inv(M)
meta=[b for b in json.load(open(A+'/volumes/brains.json'))['brains'] if b['id']=='dharani_24gw'][0]
u,v,w=meta['grid']; sp=meta['voxel_mm']
web=np.frombuffer(gzip.open(A+'/volumes/dharani_24gw.bin.gz').read(),np.uint8).reshape(w,v,u,2)
hm=web[...,0]>0
im=nib.load(R+'mri/sub-005_3d_sag_t1w_mprage.mnc'); a=np.asarray(im.dataobj,np.float32); inv=np.linalg.inv(im.affine)
mk=(np.asarray(nib.load(R+'mri/sub-005_mri_t1w_mask.mnc').dataobj)>0).astype(np.float32)
def rot(x):
    ax,ay,az=x; cx,sx,cy,sy,cz,sz=np.cos(ax),np.sin(ax),np.cos(ay),np.sin(ay),np.cos(az),np.sin(az)
    return np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
def grid(k):
    ws,vs,us=[(np.arange(n)[::k]-(n-1)/2)*sp for n in (w,v,u)]
    W,V,U=np.meshgrid(ws,vs,us,indexing='ij'); return np.stack([U.ravel(),V.ravel(),W.ravel()]),(len(ws),len(vs),len(us))
def native_ijk(q,x):
    p=(Tr@rot(x[:3]))@q+(c+x[3:6])[:,None]            # web -> stack world, rigid
    n=Mi[:3,:3]@p+Mi[:3,3:4]                          # stack world -> native world, rigid
    return inv[:3,:3]@n+inv[:3,3:4]
q,sh=grid(4); H=hm[::4,::4,::4].ravel()
def dice(x):
    m=ndi.map_coordinates(mk,native_ijk(q,x),order=1)>0.5
    return 2*(m&H).sum()/(m.sum()+H.sum())
x0=np.zeros(6); print('dice, pipeline rigid only',round(float(dice(x0)),4),flush=True)
r=minimize(lambda x:-dice(x),x0,method='Powell',options=dict(maxiter=6,xtol=1e-3,ftol=1e-4,direc=np.diag([.05]*3+[1.]*3)))
x=r.x; print('dice, rigid refined',round(float(-r.fun),4),'rot deg',np.degrees(x[:3]).round(2),'shift mm',x[3:].round(2),flush=True)
print('linear part singular values (web->native)',np.linalg.svd(Mi[:3,:3]@Tr@rot(x[:3]))[1].round(5))
out=np.zeros((w,v,u),np.float32); msk=np.zeros((w,v,u),np.float32)
U,V=np.meshgrid((np.arange(u)-(u-1)/2)*sp,(np.arange(v)-(v-1)/2)*sp)
for k in range(w):
    qq=np.stack([U.ravel(),V.ravel(),np.full(U.size,(k-(w-1)/2)*sp)]); ijk=native_ijk(qq,x)
    out[k]=ndi.map_coordinates(a,ijk,order=1).reshape(v,u); msk[k]=ndi.map_coordinates(mk,ijk,order=1).reshape(v,u)
mask=msk>0.5
print('native mask voxels lost outside the web box',round(1-mask.sum()*sp**3/(mk.sum()*.293*.293*.5),4))
print('dice full res',round(float(2*(mask&hm).sum()/(mask.sum()+hm.sum())),4))
lo,hi=np.percentile(out[mask],[2,98.5]); d=np.clip((out-lo)/(hi-lo),0,1)**0.8
depth=ndi.gaussian_filter(ndi.distance_transform_edt(mask)*sp,0.7)
r8=np.where(mask,d*254+1,0).astype(np.uint8); g=(np.clip(depth/1.6,0,1)*255).astype(np.uint8)
with gzip.open(OUT,'wb',compresslevel=9) as fo: fo.write(np.ascontiguousarray(np.stack([r8,g],-1)).tobytes())
print(OUT,os.path.getsize(OUT)//1024,'KB')
from PIL import Image
rows=[]
for sl in [(w//2,slice(None),slice(None)),(slice(None),v//2,slice(None)),(slice(None),slice(None),u//2-20)]:
    A1=web[sl+(0,)];B1=r8[sl];ck=((np.indices(A1.shape)//24).sum(0)%2).astype(bool)
    rows.append(np.hstack([A1,B1,np.where(ck,A1,B1)]))
W2=max(z.shape[1] for z in rows);img=np.vstack([np.pad(z,((0,4),(0,W2-z.shape[1]))) for z in rows])
Image.fromarray(img[::-1]).resize((img.shape[1]*2,img.shape[0]*2)).save(PNG)
