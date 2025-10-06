from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
import torch
from pc15codec.codec import encode_y as _encode_y, CodecConfig

@dataclass
class HeuristicHooks:
    topk: Optional[int] = 8
    def select_generators(self, gen_ids: Sequence[int]) -> Sequence[int]:
        if self.topk is None or self.topk >= len(gen_ids): return list(gen_ids)
        return list(gen_ids)[: self.topk]

def encode_y_with_hooks(img_y: torch.Tensor, cfg: CodecConfig, hooks: Optional[HeuristicHooks]=None) -> Dict[str, Any]:
    # Stub: intégration à venir dans la phase de recherche ; no-op pour l’instant
    return _encode_y(img_y, cfg)
