import numpy as np
import pytest

pc15 = pytest.importorskip("pc15")
CodecConfig = pytest.importorskip("pc15", reason="pc15 not installed").CodecConfig

# Import depuis le sous-package intégré
HeuristicHooks = pytest.importorskip("pc15learn.ai_hooks").HeuristicHooks
encode_y_with_hooks = pytest.importorskip("pc15learn.ai_hooks").encode_y_with_hooks


def _to_tensor(y):
    import torch
    return torch.from_numpy(y[None, None]).float()


def test_encode_with_hooks_runs():
    H = W = 64
    yy = np.linspace(-1, 1, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    yy_t = _to_tensor(yy)

    cfg = CodecConfig(tile=64, overlap=0, lambda_rd=0.015, alpha_mix=0.7, seed=111)

    # baseline encode
    enc0 = pc15.encode_y(yy_t, cfg)

    # hooks heuristiques (API actuelle : topk, etc.)
    hooks = HeuristicHooks(topk=8)
    enc1 = encode_y_with_hooks(yy_t, cfg, hooks)

    assert isinstance(enc0, dict) and isinstance(enc1, dict)
    assert isinstance(enc0.get("bitstream", b""), (bytes, bytearray))
    assert isinstance(enc1.get("bitstream", b""), (bytes, bytearray))
    assert len(enc0["bitstream"]) > 0 and len(enc1["bitstream"]) > 0
