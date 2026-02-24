from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import nn

Batch = Tuple[torch.Tensor, torch.Tensor]
MetricFn = Callable[[nn.Module, Batch], float]


def metric_improved(new_metric: float, old_metric: float, min_delta: float = 0.0) -> bool:
    """Higher metric is better."""
    return new_metric > old_metric + min_delta


def negative_mse_metric(model: nn.Module, batch: Batch) -> float:
    """Default metric useful for no-label-shift toy setups."""
    inputs, targets = batch
    with torch.no_grad():
        predictions = model(inputs)
        return float(-(torch.nn.functional.mse_loss(predictions, targets)))
