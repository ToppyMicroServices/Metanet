"""Self-supervised proxy losses for Safe MetaNet.

These losses operate on **model outputs** (logits) alone, without any labels,
making them suitable for test-time adaptation.

Available losses
----------------
entropy_loss
    Minimise the Shannon entropy of the softmax distribution.
    Low entropy ↔ confident predictions ↔ better-adapted model.
    This is the standard TTA proxy (Wang et al., 2021 – Tent).

reconstruction_loss
    Dummy MSE reconstruction loss: the model is asked to reconstruct its
    own input from its output.  Useful as a placeholder when the backbone
    output dimension equals the input dimension; otherwise falls back to
    entropy.

The :func:`compute_proxy_metric` function returns a *lower-is-better*
scalar that the Safe MetaNet loop uses to decide accept / reject.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the softmax distribution (lower = more confident).

    Parameters
    ----------
    logits:
        Raw model outputs of shape ``(B, C)`` where *C* is the number of
        classes.

    Returns
    -------
    torch.Tensor
        Scalar mean entropy across the batch.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    ent = -(probs * log_probs).sum(dim=-1)   # (B,)
    return ent.mean()


def reconstruction_loss(
    logits: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """MSE between model output and input (dummy self-supervised signal).

    Used when ``output_dim == input_dim``.  Falls back to
    :func:`entropy_loss` otherwise.

    Parameters
    ----------
    logits:
        Raw model outputs of shape ``(B, D_out)``.
    inputs:
        Model inputs of shape ``(B, D_in)``.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    if logits.shape[-1] != inputs.shape[-1]:
        return entropy_loss(logits)
    return F.mse_loss(logits, inputs.detach())


# ---------------------------------------------------------------------------
# Unified proxy-metric API
# ---------------------------------------------------------------------------

def compute_proxy_metric(
    model: nn.Module,
    x: torch.Tensor,
    loss_type: str = "entropy",
) -> torch.Tensor:
    """Compute the proxy metric for a batch *x* using *model*.

    The metric is **lower-is-better**: Safe MetaNet accepts an update when
    ``metric_after < metric_before - threshold``.

    Parameters
    ----------
    model:
        The backbone (with adapters injected).
    x:
        Input batch of shape ``(B, ...)``.
    loss_type:
        ``"entropy"`` (default) or ``"reconstruction"``.

    Returns
    -------
    torch.Tensor
        Scalar proxy loss (no grad).
    """
    with torch.no_grad():
        logits = model(x)

    if loss_type == "entropy":
        return entropy_loss(logits)
    if loss_type == "reconstruction":
        return reconstruction_loss(logits, x.flatten(start_dim=1))
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


def differentiable_proxy_loss(
    model: nn.Module,
    x: torch.Tensor,
    loss_type: str = "entropy",
    trust_region_lambda: float = 0.0,
    adapter_params: list | None = None,
) -> torch.Tensor:
    """Differentiable version of the proxy loss used for gradient updates.

    Optionally adds an L2 trust-region penalty on adapter parameters.

    Parameters
    ----------
    model:
        The backbone with adapters.
    x:
        Input batch.
    loss_type:
        ``"entropy"`` or ``"reconstruction"``.
    trust_region_lambda:
        Weight for the L2 penalty on adapter parameters.
    adapter_params:
        List of adapter :class:`~torch.nn.Parameter` tensors.  Required when
        *trust_region_lambda* > 0.

    Returns
    -------
    torch.Tensor
        Scalar loss with gradient.
    """
    logits = model(x)

    if loss_type == "entropy":
        loss = entropy_loss(logits)
    elif loss_type == "reconstruction":
        loss = reconstruction_loss(logits, x.flatten(start_dim=1))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    if trust_region_lambda > 0.0 and adapter_params:
        penalty = sum(p.pow(2).sum() for p in adapter_params)
        loss = loss + trust_region_lambda * penalty

    return loss
