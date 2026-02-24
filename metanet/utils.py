from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn


def snapshot_parameters(modules: Iterable[nn.Module]) -> Dict[str, torch.Tensor]:
    """Clone parameters for rollback."""
    return {
        f"{id(module)}:{name}": parameter.detach().clone()
        for module in modules
        for name, parameter in module.named_parameters()
    }


def restore_parameters(modules: Iterable[nn.Module], snapshot: Dict[str, torch.Tensor]) -> None:
    """Restore parameters from a previous snapshot."""
    with torch.no_grad():
        for module in modules:
            for name, parameter in module.named_parameters():
                key = f"{id(module)}:{name}"
                if key in snapshot:
                    parameter.copy_(snapshot[key])


def set_trainable_scope(modules: Iterable[nn.Module], scope_size: int) -> None:
    """Enable gradients only for modules in [0, scope_size)."""
    for index, module in enumerate(modules):
        trainable = index < scope_size
        for parameter in module.parameters():
            parameter.requires_grad = trainable
