from __future__ import annotations
import torch, math
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
class HexGrid(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="HEX_GRID",
            param_specs=(ParamSpec("cells","int",(2,256),"per img",2.0),
                         ParamSpec("thick","float",(0.003,0.2),None,0.005)),supports_noise=False)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B= params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        cells=params[:,0].view(B,1,1).to(dtype); thick=params[:,1].view(B,1,1).to(dtype)
        s=cells/2.0; x=(xx+1.0)*s; y=(yy+1.0)*s*math.sqrt(3)/2.0
        q=x - y/math.sqrt(3); r=y*2.0/math.sqrt(3)
        fq=q - torch.floor(q); fr=r - torch.floor(r); fs=(-fq-fr)%1.0
        d=torch.minimum(torch.minimum(fq,fr),fs)
        v=(d<thick).to(dtype); return (v*2.0-1.0).unsqueeze(1)
GEN=HexGrid()
