from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class QVTable:
    id: int
    name: str
    centers: torch.Tensor  # [K,P]
    scales: torch.Tensor   # [P]

class QVCodec:
    def __init__(self, tables: dict[int, QVTable]):
        self.tables = tables

    def quantize(self, params: torch.Tensor, table_id: int):
        tab = self.tables[table_id]
        d = (params[None, :, :] - tab.centers[:, None, :]).pow(2).sum(-1)  # [K,B]
        idx = d.argmin(dim=0)
        centers = tab.centers[idx]
        offsets = (params - centers) / tab.scales
        return idx, offsets

    def dequantize(self, idx: torch.Tensor, offsets: torch.Tensor, table_id: int):
        tab = self.tables[table_id]
        centers = tab.centers[idx]
        return centers + offsets * tab.scales
