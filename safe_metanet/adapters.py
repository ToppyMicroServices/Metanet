"""Lightweight adapter modules for Safe MetaNet.

Two adapter flavours are provided:

* :class:`LoRALinear`       – wraps a frozen :class:`~torch.nn.Linear` with a
  low-rank side-path  ``W += B @ A``  (Hu et al., 2022).
* :class:`LinearAdapter`    – wraps a frozen :class:`~torch.nn.Linear` with a
  small bottleneck  ``y += W_up(act(W_down(x)))``.

Both keep the original weight **frozen** and only train the small adapter
parameters, which is the contract required by the Safe MetaNet loop.

:func:`inject_adapters` provides a convenience wrapper that replaces
the ``layers`` entries of a :class:`~safe_metanet.backbone.ToyMLP` /
:class:`~safe_metanet.backbone.ToyCNN` with adapter-wrapped versions.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import SafeMetaNetConfig


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adaptation of a frozen :class:`~torch.nn.Linear` layer.

    The effective weight is  ``W_frozen + scale * B @ A``  where ``A`` and
    ``B`` are rank-*r* matrices initialised so that the delta is zero at
    the start of adaptation (``B`` initialised to zero).

    Parameters
    ----------
    base_layer:
        The frozen linear layer to wrap.
    rank:
        Rank of the low-rank decomposition.
    scale:
        Scaling factor (``alpha / rank`` in the LoRA paper; default 1.0).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Keep the frozen base weight
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scale = scale

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.base_layer.in_features}, "
            f"out={self.base_layer.out_features}, "
            f"rank={self.rank}"
        )


# ---------------------------------------------------------------------------
# Linear Adapter (bottleneck)
# ---------------------------------------------------------------------------

class LinearAdapter(nn.Module):
    """Bottleneck residual adapter for a frozen :class:`~torch.nn.Linear`.

    Adds a residual path  ``y += W_up(ReLU(W_down(x)))``  where
    ``W_down : d_in → bottleneck``  and  ``W_up : bottleneck → d_out``.
    Initialised to output zero (``W_up`` zeroed) so adaptation starts from
    the pretrained model.

    Parameters
    ----------
    base_layer:
        The frozen linear layer to wrap.
    bottleneck:
        Hidden size of the adapter bottleneck.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        bottleneck: int = 8,
    ) -> None:
        super().__init__()
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.down = nn.Linear(in_features, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, out_features, bias=False)
        self.act = nn.ReLU()

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        adapter_out = self.up(self.act(self.down(x)))
        return base_out + adapter_out

    def extra_repr(self) -> str:
        return (
            f"in={self.base_layer.in_features}, "
            f"out={self.base_layer.out_features}, "
            f"bottleneck={self.down.out_features}"
        )


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def wrap_linear(
    layer: nn.Linear,
    adapter_type: str,
    rank: int = 4,
    bottleneck: int = 8,
) -> nn.Module:
    """Wrap a single :class:`~torch.nn.Linear` with an adapter.

    Parameters
    ----------
    layer:
        The linear layer to wrap.
    adapter_type:
        ``"lora"`` or ``"linear_adapter"``.
    rank:
        LoRA rank (used when *adapter_type* == ``"lora"``).
    bottleneck:
        Adapter bottleneck size (used when *adapter_type* == ``"linear_adapter"``).
    """
    if adapter_type == "lora":
        return LoRALinear(layer, rank=rank)
    if adapter_type == "linear_adapter":
        return LinearAdapter(layer, bottleneck=bottleneck)
    raise ValueError(f"Unknown adapter type: {adapter_type!r}")


def inject_adapters(
    model: nn.Module,
    cfg: SafeMetaNetConfig,
    layer_indices: Optional[List[int]] = None,
) -> None:
    """Inject adapters into ``model.layers`` in-place.

    Only layers whose index is in *layer_indices* receive an adapter.
    If *layer_indices* is ``None``, only the *last* layer is adapted
    (minimal initial scope).

    Parameters
    ----------
    model:
        A backbone with a ``layers`` :class:`~torch.nn.ModuleList`.
    cfg:
        Configuration used to pick adapter type and hyperparameters.
    layer_indices:
        Indices of ``model.layers`` to wrap.  ``None`` → last layer only.
    """
    n = len(model.layers)
    if layer_indices is None:
        layer_indices = [n - 1]

    for idx in layer_indices:
        orig = model.layers[idx]
        if isinstance(orig, nn.Linear):
            model.layers[idx] = wrap_linear(
                orig,
                adapter_type=cfg.adapter_type,
                rank=cfg.lora_rank,
                bottleneck=cfg.adapter_bottleneck,
            )
        # Conv2d layers are left as-is (not adapted in this minimal prototype).


def adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return only the trainable adapter parameters from *model*."""
    params = []
    for module in model.modules():
        if isinstance(module, (LoRALinear, LinearAdapter)):
            params.extend(
                p for p in module.parameters() if p.requires_grad
            )
    return params


def snapshot_adapters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a deep-copy snapshot of all adapter parameter tensors."""
    snap: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LinearAdapter)):
            for pname, p in module.named_parameters():
                if p.requires_grad:
                    snap[f"{name}.{pname}"] = p.data.clone()
    return snap


def restore_adapters(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    """Restore adapter parameters from a snapshot produced by
    :func:`snapshot_adapters`."""
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LinearAdapter)):
            for pname, p in module.named_parameters():
                if p.requires_grad:
                    key = f"{name}.{pname}"
                    if key in snapshot:
                        p.data.copy_(snapshot[key])
