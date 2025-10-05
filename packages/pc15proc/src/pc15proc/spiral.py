from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
class Spiral(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="SPIRAL",
            param_specs=(ParamSpec("turns","float",(0.5,32.0),None,0.5),
                         ParamSpec("tight","float",(0.1,5.0),None,0.1)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B= params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        turns=params[:,0].view(B,1,1).to(dtype); tight=params[:,1].view(B,1,1).to(dtype)
        theta=torch.atan2(yy,xx)+torch.pi; r=torch.sqrt(xx*xx+yy*yy)+1e-6
        s=torch.sin(tight*r + turns*theta); return s.clamp(-1,1).unsqueeze(1)
GEN=Spiral()
