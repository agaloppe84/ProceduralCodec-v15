from __future__ import annotations
import os
from typing import Any

def get_device(strict_gpu: bool = True):
    try:
        import torch
    except Exception as e:
        from .errors import MissingCudaError
        raise MissingCudaError(f"Torch unavailable: {e}")
    allow_cpu = os.getenv("PC15_ALLOW_CPU_TESTS", "0") == "1"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if strict_gpu and not allow_cpu:
        from .errors import MissingCudaError
        raise MissingCudaError("Manque: GPU CUDA")
    return torch.device("cpu")

def cuda_info() -> dict[str, Any]:
    try:
        import torch
        return {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as e:
        return {"error": str(e)}
