from __future__ import annotations
import torch, math
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
class TriTiling(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="TRI_TILING",
            param_specs=(ParamSpec("cells","int",(2,256),"per img",2.0),
                         ParamSpec("line","float",(0.003,0.2),None,0.005)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B= params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        cells=params[:,0].view(B,1,1).to(dtype); line=params[:,1].view(B,1,1).to(dtype)
        u=(xx+1.0)*cells*0.5; v=(yy+1.0)*cells*0.5*math.sqrt(3)/2.0
        a=torch.abs((u - torch.floor(u)) - 0.5); b=torch.abs((v - torch.floor(v)) - 0.5)
        d=torch.minimum(a,b); return ((d<line).to(dtype)*2.0-1.0).unsqueeze(1)
GEN=TriTiling()
