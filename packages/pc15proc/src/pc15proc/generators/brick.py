from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
class Brick(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="BRICK",
            param_specs=(ParamSpec("rows","int",(2,512),"per img",2.0),
                         ParamSpec("cols","int",(2,512),"per img",2.0),
                         ParamSpec("mortar","float",(0.002,0.1),None,0.002)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        rows=params[:,0].view(B,1,1).to(dtype); cols=params[:,1].view(B,1,1).to(dtype); mortar=params[:,2].view(B,1,1).to(dtype)
        fy=(yy+1.0)*rows*0.5; fx=(xx+1.0)*cols*0.5
        yi=torch.floor(fy); xi=torch.floor(fx)
        fx_shift=fx + ((yi.to(torch.int64) & 1).to(dtype) * 0.5)
        xi2=torch.floor(fx_shift)
        dx=torch.abs(fx_shift - xi2 - 0.5); dy=torch.abs(fy - yi - 0.5)
        v=((dx>mortar) & (dy>mortar)).to(dtype)
        return (v*2.0-1.0).unsqueeze(1)
GEN=Brick()
