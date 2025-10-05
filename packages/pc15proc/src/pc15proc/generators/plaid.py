from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
class Plaid(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="PLAID",
            param_specs=(ParamSpec("fx","float",(0.5,64.0),"cycles/img",0.5),
                         ParamSpec("fy","float",(0.5,64.0),"cycles/img",0.5),
                         ParamSpec("amp","float",(0.1,1.0),None,0.1),
                         ParamSpec("phase","float",(0.0,6.28318),"rad",0.2)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B= params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        fx=params[:,0].view(B,1,1).to(dtype); fy=params[:,1].view(B,1,1).to(dtype)
        a=params[:,2].view(B,1,1).to(dtype); ph=params[:,3].view(B,1,1).to(dtype)
        s=torch.sin(2*torch.pi*fx*xx+ph)+torch.sin(2*torch.pi*fy*yy+ph)
        return (s*0.5*a).clamp(-1,1).unsqueeze(1)
GEN = Plaid()
