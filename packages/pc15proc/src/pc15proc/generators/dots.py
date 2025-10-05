from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
class Dots(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="DOTS",
            param_specs=(ParamSpec("cells","int",(2,256),"per img",2.0),
                         ParamSpec("radius","float",(0.01,0.5),None,0.01)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B= params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        cells=params[:,0].view(B,1,1).to(dtype); rad=params[:,1].view(B,1,1).to(dtype)
        fx=(xx+1.0)*cells*0.5; fy=(yy+1.0)*cells*0.5
        cx=torch.floor(fx)+0.5; cy=torch.floor(fy)+0.5
        r=torch.sqrt((fx-cx)**2+(fy-cy)**2)
        v=(r<=rad).to(dtype); return (v*2.0-1.0).unsqueeze(1)
GEN=Dots()
