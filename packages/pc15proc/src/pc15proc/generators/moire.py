from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
class Moire(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="MOIRE",
            param_specs=(ParamSpec("f1","float",(0.5,64.0),"cycles/img",0.5),
                         ParamSpec("f2","float",(0.5,64.0),"cycles/img",0.5),
                         ParamSpec("delta_deg","float",(-10.0,10.0),"deg",0.5)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B= params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        f1=params[:,0].view(B,1,1).to(dtype); f2=params[:,1].view(B,1,1).to(dtype)
        d =torch.deg2rad(params[:,2]).view(B,1,1).to(dtype)
        s1=torch.sin(2*torch.pi*(f1*xx)); xr=xx*torch.cos(d)-yy*torch.sin(d)
        s2=torch.sin(2*torch.pi*(f2*xr)); return (0.5*(s1+s2)).clamp(-1,1).unsqueeze(1)
GEN=Moire()
