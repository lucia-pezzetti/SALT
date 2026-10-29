"""Small helpers for explicit, consistent PyTorch device handling."""
from __future__ import annotations

import torch
import torch.nn as nn


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` and fail clearly when CUDA was explicitly requested."""
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Check the container's PyTorch build and the Slurm GPU allocation."
        )
    if value not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported torch device '{requested}'. Use auto, cpu, or cuda.")
    return torch.device(value)


def seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def from_numpy(array, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device)


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")
