from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(eq=True)
class TileRec:
    """Enregistrement de tuile (Step 1, version simple).
    Note: pas encore d'encodage rANS ici (payload=b"" possible).
    """
    tile_id: int
    gen_id: int
    qv_id: int
    seed: int
    rec_flags: int = 0
    payload_fmt: int = 0
    payload: bytes = field(default_factory=bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": int(self.tile_id),
            "gen_id": int(self.gen_id),
            "qv_id": int(self.qv_id),
            "seed": int(self.seed),
            "rec_flags": int(self.rec_flags),
            "payload_fmt": int(self.payload_fmt),
            "payload": bytes(self.payload),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TileRec":
        return TileRec(
            tile_id=int(d["tile_id"]),
            gen_id=int(d["gen_id"]),
            qv_id=int(d["qv_id"]),
            seed=int(d["seed"]),
            rec_flags=int(d.get("rec_flags", 0)),
            payload_fmt=int(d.get("payload_fmt", 0)),
            payload=bytes(d.get("payload", b"")),
        )
