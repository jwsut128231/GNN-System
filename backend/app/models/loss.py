"""Shared loss helpers used by the model layer.

Centralised here so the multi-target weighted MSE definition stays consistent
across GCN/GAT/SAGE/GIN/MLP and the heterogeneous wrapper.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def weighted_regression_loss(
    out: torch.Tensor,
    y: torch.Tensor,
    loss_weights: Optional[torch.Tensor],
    num_targets: int,
) -> torch.Tensor:
    """Compute MSE with optional per-target weighting.

    * Single-target (``num_targets == 1``): standard ``F.mse_loss``. Shapes for
      ``out`` and ``y`` are both either ``[*]`` or ``[*, 1]``.
    * Multi-target (``num_targets > 1``): shape is ``[*, T]``. Per-target
      weights are applied before averaging. Final scalar = mean over batch
      dimensions of (sum across targets of weighted squared errors).
    """
    if num_targets <= 1 or out.dim() == 1 or out.shape[-1] == 1:
        return F.mse_loss(out.reshape_as(y), y)

    se = (out - y) ** 2  # shape [*, T]
    if loss_weights is not None:
        w = loss_weights.to(out.device).to(out.dtype)
        if w.dim() == 1:
            se = se * w  # broadcasts over leading dims
    return se.sum(dim=-1).mean()
