"""Device selection, shared by every script that renders or trains.

CUDA, then Intel XPU, then CPU. Detection runs once and prints what it chose,
so a run that silently fell back to CPU is visible in the log.
"""

import torch

_cache = None


def get_device() -> torch.device:
    global _cache
    if _cache is not None:
        return _cache
    if torch.cuda.is_available():
        _cache = torch.device("cuda")
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        _cache = torch.device("xpu")
        print(f"Using device: xpu ({torch.xpu.get_device_name(0)})")
    else:
        _cache = torch.device("cpu")
        print("Using device: cpu")
    return _cache
