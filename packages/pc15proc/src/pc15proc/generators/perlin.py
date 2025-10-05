from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from .noise import perlin2d
class Perlin(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="PERLIN",
            param_specs=(ParamSpec("scale","float",(1.0,256.0),"cells/img",1.0),
                         ParamSpec("octaves","int",(1,6),None,1.0),
                         ParamSpec("persistence","float",(0.3,1.0),None,0.1)),supports_noise=True)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        base=params[:,0].view(B,1,1).to(dtype); octv=params[:,1].view(B,1,1).to(torch.int64)
        pers=params[:,2].view(B,1,1).to(dtype); seed64=seeds.view(B,1,1).to(torch.int64)
        val=torch.zeros((B,h,w),device=device,dtype=dtype); amp=torch.ones((B,1,1),device=device,dtype=dtype); freq=base.clone()
        for k in range(6):
            use=(k<octv).to(dtype); v=perlin2d(xx,yy,freq,seed64+k)
            val=val+v*amp*use; amp=amp*pers; freq=freq*2.0
        sum_amp=(1.0 - torch.pow(pers, octv.to(dtype)))/(1.0 - pers + 1e-6)
        return (val/(sum_amp+1e-6)).clamp(-1,1).unsqueeze(1)
GEN=Perlin()
