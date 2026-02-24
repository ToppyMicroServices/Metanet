"""Core Safe MetaNet loop.

The :class:`SafeMetaNet` class implements the test-time weight editing
algorithm described below.

Algorithm (one test sample)
---------------------------
1.  Compute proxy metric  **M_before**  on the current sample with the
    current adapter state.
2.  Take *num_steps* gradient steps on the **editable** adapter parameters
    using the differentiable proxy loss + trust-region penalty.
3.  Compute  **M_after**  with the updated parameters.
4.  **Accept** if ``M_after < M_before - threshold``; keep the update.
5.  **Reject** otherwise; restore the adapter snapshot.
    Unless :attr:`disable_scope_expansion`, add the next layer to the
    editable scope and retry (up to ``len(expansion_schedule)`` levels).
6.  Log the outcome via :class:`~safe_metanet.logger.SafeMetaNetLogger`.

Ablation switches (from :class:`~safe_metanet.config.SafeMetaNetConfig`)
------------------------------------------------------------------------
* ``disable_rollback``         – always accept updates (no rollback).
* ``disable_scope_expansion``  – never expand beyond the initial scope.
"""

from __future__ import annotations

import copy
import logging
from typing import Iterator, List, Optional

import torch
import torch.nn as nn

from .adapters import (
    adapter_parameters,
    inject_adapters,
    restore_adapters,
    snapshot_adapters,
)
from .config import SafeMetaNetConfig
from .logger import SafeMetaNetLogger, UpdateRecord
from .loss import compute_proxy_metric, differentiable_proxy_loss


class SafeMetaNet:
    """Test-time weight editing with adaptive scope expansion and rollback.

    Parameters
    ----------
    model:
        A backbone (already built and frozen).  Adapters will be injected
        in-place during initialisation.
    cfg:
        Configuration dataclass.
    logger:
        Optional logger; a default one is created if ``None``.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: SafeMetaNetConfig,
        logger: Optional[SafeMetaNetLogger] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.logger = logger or SafeMetaNetLogger(level=cfg.log_level)

        # Expansion schedule: ordered list of layer names.
        # We start with the first entry (minimal scope) and accumulate.
        self._schedule: List[str] = list(cfg.expansion_schedule)
        # Current scope is the set of layer names whose adapters are editable.
        self._scope: List[str] = [self._schedule[0]] if self._schedule else []
        # How many schedule entries are currently active (index pointer).
        self._scope_ptr: int = 1

        # Inject adapters for the initial scope.
        self._layer_name_to_idx = self._build_layer_index()
        self._inject_for_scope(self._scope)

        self._step: int = 0

    # ------------------------------------------------------------------ setup

    def _build_layer_index(self) -> dict:
        """Map layer names (``"layer_0"`` etc.) to indices in ``model.layers``."""
        index: dict = {}
        if hasattr(self.model, "layers"):
            for i in range(len(self.model.layers)):
                index[f"layer_{i}"] = i
        return index

    def _inject_for_scope(self, scope: List[str]) -> None:
        """Inject (or re-inject) adapters for all layers in *scope*."""
        indices = [
            self._layer_name_to_idx[name]
            for name in scope
            if name in self._layer_name_to_idx
        ]
        inject_adapters(self.model, self.cfg, layer_indices=indices)

    # ------------------------------------------------------------------ step

    def step(self, x: torch.Tensor, sample_idx: int = 0) -> dict:
        """Process one test sample through the Safe MetaNet loop.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, ...)``.
        sample_idx:
            Index of this sample (for logging).

        Returns
        -------
        dict
            Keys: ``action``, ``metric_before``, ``metric_after``,
            ``scope``, ``scope_expanded``.
        """
        self._step += 1
        scope_expanded = False

        # We may retry with an expanded scope on rejection.
        for attempt in range(len(self._schedule) + 1):
            snap = snapshot_adapters(self.model)
            metric_before = compute_proxy_metric(
                self.model, x, loss_type="entropy"
            ).item()

            # Gradient update(s) on editable adapter parameters only.
            params = adapter_parameters(self.model)
            if params:
                opt = torch.optim.Adam(params, lr=self.cfg.lr)
                for _ in range(self.cfg.num_steps):
                    opt.zero_grad()
                    loss = differentiable_proxy_loss(
                        self.model,
                        x,
                        loss_type="entropy",
                        trust_region_lambda=self.cfg.trust_region_lambda,
                        adapter_params=params,
                    )
                    loss.backward()
                    opt.step()

            metric_after = compute_proxy_metric(
                self.model, x, loss_type="entropy"
            ).item()

            improved = metric_after < metric_before - self.cfg.rollback_threshold

            if self.cfg.disable_rollback or improved:
                action = "accept"
                record = UpdateRecord(
                    step=self._step,
                    sample_idx=sample_idx,
                    action=action,
                    metric_before=metric_before,
                    metric_after=metric_after,
                    scope=list(self._scope),
                    scope_expanded=scope_expanded,
                    gradient_steps=self.cfg.num_steps if params else 0,
                )
                self.logger.log_update(record)
                return {
                    "action": action,
                    "metric_before": metric_before,
                    "metric_after": metric_after,
                    "scope": list(self._scope),
                    "scope_expanded": scope_expanded,
                }

            # Reject: rollback parameters.
            restore_adapters(self.model, snap)
            action = "reject"

            # Expand scope if allowed and schedule has more entries.
            if (
                not self.cfg.disable_scope_expansion
                and self._scope_ptr < len(self._schedule)
            ):
                new_layer = self._schedule[self._scope_ptr]
                self._scope.append(new_layer)
                self._scope_ptr += 1
                self._inject_for_scope([new_layer])
                scope_expanded = True
            else:
                # No more expansion; log the rejection and return.
                record = UpdateRecord(
                    step=self._step,
                    sample_idx=sample_idx,
                    action=action,
                    metric_before=metric_before,
                    metric_after=metric_after,
                    scope=list(self._scope),
                    scope_expanded=False,
                    gradient_steps=self.cfg.num_steps if params else 0,
                )
                self.logger.log_update(record)
                return {
                    "action": action,
                    "metric_before": metric_before,
                    "metric_after": metric_after,
                    "scope": list(self._scope),
                    "scope_expanded": False,
                }

        # Fallback (should not reach here): log the final rejection.
        record = UpdateRecord(
            step=self._step,
            sample_idx=sample_idx,
            action="reject",
            metric_before=metric_before,
            metric_after=metric_after,
            scope=list(self._scope),
            scope_expanded=scope_expanded,
            gradient_steps=0,
        )
        self.logger.log_update(record)
        return {
            "action": "reject",
            "metric_before": metric_before,
            "metric_after": metric_after,
            "scope": list(self._scope),
            "scope_expanded": scope_expanded,
        }

    # ------------------------------------------------------------------ loop

    def run(self, data_iter: Iterator[torch.Tensor]) -> None:
        """Run the Safe MetaNet loop over an iterable of batches.

        Parameters
        ----------
        data_iter:
            An iterator / iterable of input tensors, one per test step.
        """
        for idx, x in enumerate(data_iter):
            self.step(x, sample_idx=idx)

    # ------------------------------------------------------------------ info

    @property
    def current_scope(self) -> List[str]:
        """The set of layer names currently being adapted."""
        return list(self._scope)

    def summary(self) -> str:
        """Return a human-readable summary from the logger."""
        return self.logger.summary()
