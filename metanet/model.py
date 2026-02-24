from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class LoRAAdapter(nn.Module):
    """Minimal LoRA-style adapter for Linear layers."""

    is_adapter = True

    def __init__(self, in_features: int, out_features: int, rank: int = 2) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SafeMetaNetModel(nn.Module):
    """Frozen base model + editable adapters."""

    def __init__(self, base_layers: Iterable[nn.Linear], rank: int = 2) -> None:
        super().__init__()
        self.base_layers = nn.ModuleList(list(base_layers))
        self.adapters = nn.ModuleList(
            [LoRAAdapter(layer.in_features, layer.out_features, rank=rank) for layer in self.base_layers]
        )
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for parameter in self.base_layers.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.base_layers):
            x = layer(x) + self.adapters[index](x)
            if index < len(self.base_layers) - 1:
                x = torch.relu(x)
        return x

    def adapter_modules(self) -> List[nn.Module]:
        return list(self.adapters)
