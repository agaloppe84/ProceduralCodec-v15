from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from .noise import worley_f1
class Worley(Generator):
    @property
    def info(self)->GeneratorInfo:
        return GeneratorInfo(name="WORLEY",
            param_specs=(ParamSpec("scale","float",(1.0,256.0),"cells/img",1.0),
                         ParamSpec("metric","enum",None,("euclidean","manhattan","chebyshev")),
                         ParamSpec("invert","bool",None,None)),supports_noise=True)
    @torch.no_grad()
    def render(self,tiles_hw,params,seeds,*,device,dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        scale=params[:,0].view(B,1,1).to(dtype); mindex=torch.clamp(params[:,1],0,2).to(torch.int64).view(B,1,1)
        invert=(params[:,2]>0.5).view(B,1,1); metrics=["euclidean","manhattan","chebyshev"]
        outs=[]
        for i in range(B):
            metric=metrics[int(mindex[i,0,0].item())]
            v=worley_f1(xx,yy,scale[i:i+1],seeds[i:i+1].view(1,1,1).to(torch.int64),metric=metric)
            outs.append(v)
        v=torch.cat(outs,dim=0)
        v=torch.where(invert,1.0 - v, v)
        return (v*2.0-1.0).clamp(-1,1).unsqueeze(1)
GEN=Worley()
