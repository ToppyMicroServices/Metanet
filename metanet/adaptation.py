from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from .metrics import Batch, MetricFn, metric_improved
from .utils import restore_parameters, set_trainable_scope, snapshot_parameters


@dataclass
class StepResult:
    accepted: bool
    metric_before: float
    metric_after: float
    active_scope: int


class SafeMetaNetLoop:
    """Core Safe MetaNet accept/reject loop with adaptive scope expansion."""

    def __init__(
        self,
        model: nn.Module,
        metric_fn: MetricFn,
        loss_fn: nn.Module | None = None,
        learning_rate: float = 1e-3,
        min_delta: float = 0.0,
    ) -> None:
        self.model = model
        self.metric_fn = metric_fn
        self.loss_fn = loss_fn or nn.MSELoss()
        self.learning_rate = learning_rate
        self.min_delta = min_delta

        self.adapters: List[nn.Module] = model.adapter_modules()
        self.active_scope = 1
        self._configure_scope()

    def _configure_scope(self) -> None:
        set_trainable_scope(self.adapters, self.active_scope)
        trainable_parameters = [
            parameter
            for adapter in self.adapters[: self.active_scope]
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.SGD(trainable_parameters, lr=self.learning_rate)

    def _expand_scope(self) -> None:
        if self.active_scope < len(self.adapters):
            self.active_scope += 1
            self._configure_scope()

    def step(self, batch: Batch) -> StepResult:
        metric_before = self.metric_fn(self.model, batch)
        snapshot = snapshot_parameters(self.adapters[: self.active_scope])

        self.model.train()
        self.optimizer.zero_grad()
        inputs, targets = batch
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()

        metric_after = self.metric_fn(self.model, batch)
        accepted = metric_improved(metric_after, metric_before, self.min_delta)

        if not accepted:
            restore_parameters(self.adapters[: self.active_scope], snapshot)
            self._expand_scope()

        return StepResult(
            accepted=accepted,
            metric_before=metric_before,
            metric_after=metric_after,
            active_scope=self.active_scope,
        )
