from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import perlin2d
class Wood(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="WOOD",
            param_specs=(ParamSpec("rings","float",(2.0,128.0),"cycles/img",1.0),
                         ParamSpec("noise_scale","float",(2.0,128.0),"cells/img",1.0),
                         ParamSpec("noise_amp","float",(0.0,1.0),None,0.05)),supports_noise=True)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        rings=params[:,0].view(B,1,1).to(dtype); ns=params[:,1].view(B,1,1).to(dtype); na=params[:,2].view(B,1,1).to(dtype)
        r=torch.sqrt(xx*xx+yy*yy); n=perlin2d(xx,yy,ns,seeds.view(B,1,1).to(torch.int64))
        s=torch.sin(2*torch.pi*rings*(r + na*n))
        return s.clamp(-1,1).unsqueeze(1)
GEN=Wood()
