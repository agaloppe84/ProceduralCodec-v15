from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
class Sunburst(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="SUNBURST",
            param_specs=(ParamSpec("rays","int",(2,256),"count",2.0),
                         ParamSpec("gamma","float",(0.5,3.0),None,0.1)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        rays=params[:,0].view(B,1,1).to(dtype); gamma=params[:,1].view(B,1,1).to(dtype)
        theta=torch.atan2(yy,xx)+torch.pi
        t=(theta/(2*torch.pi))*rays
        frac=t - torch.floor(t)
        v=1.0 - torch.pow(torch.abs(frac-0.5)*2.0, gamma)
        return (v*2.0-1.0).clamp(-1,1).unsqueeze(1)
GEN=Sunburst()
