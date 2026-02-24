"""Toy backbone models for Safe MetaNet experiments.

The backbones are intentionally small so that the algorithm is the focus,
not the model capacity.  Two variants are provided:

* :class:`ToyMLP`  – a stack of fully-connected layers.
* :class:`ToyCNN`  – a shallow convolutional network for 2-D inputs.

Both backbones:
 - expose their layers through a ``layers`` :class:`~torch.nn.ModuleList`
   so that :mod:`safe_metanet.adapters` can wrap individual layers.
 - are fully frozen by default; call :func:`freeze_backbone` to enforce this.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .config import SafeMetaNetConfig


class ToyMLP(nn.Module):
    """Simple multi-layer perceptron.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dim:
        Width of each hidden layer.
    output_dim:
        Number of output logits.
    num_layers:
        Total number of linear layers (including the output layer).
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 10,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"

        dims: List[int] = (
            [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        )
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B, output_dim)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x

    # ------------------------------------------------------------------
    def layer_names(self) -> List[str]:
        """Return canonical layer names used in expansion schedules."""
        return [f"layer_{i}" for i in range(len(self.layers))]


class ToyCNN(nn.Module):
    """Shallow CNN for 1-channel 2-D inputs (default: 8×8 patches).

    The network flattens after two conv layers and passes through a linear
    classifier, so it has the same ``layers`` interface as :class:`ToyMLP`.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    hidden_dim:
        Number of feature maps in conv layers; width of the hidden linear layer.
    output_dim:
        Number of output logits.
    spatial_size:
        Height (= width) of the input spatial grid.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 10,
        spatial_size: int = 8,
    ) -> None:
        super().__init__()
        # Two 3×3 conv layers with 'same' padding keep spatial dims constant.
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        flat = hidden_dim * spatial_size * spatial_size
        self.fc = nn.Linear(flat, output_dim)
        self.act = nn.ReLU()

        # Expose as layers for adapter injection
        self.layers = nn.ModuleList([self.conv1, self.conv2, self.fc])
        self._spatial_size = spatial_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, output_dim)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

    def layer_names(self) -> List[str]:
        return [f"layer_{i}" for i in range(len(self.layers))]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_backbone(cfg: SafeMetaNetConfig) -> nn.Module:
    """Instantiate and freeze a backbone from *cfg*."""
    if cfg.backbone == "mlp":
        model = ToyMLP(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            num_layers=cfg.num_layers,
        )
    elif cfg.backbone == "cnn":
        model = ToyCNN(
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
        )
    else:
        raise ValueError(f"Unknown backbone type: {cfg.backbone!r}")

    freeze_backbone(model)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters in *model* in-place."""
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_layer(layer: nn.Module) -> None:
    """Un-freeze all parameters of a single *layer*."""
    for p in layer.parameters():
        p.requires_grad_(True)
