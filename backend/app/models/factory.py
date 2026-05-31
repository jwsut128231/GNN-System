"""Model factory — returns the right Lightning module for the task.

Two code paths:
    * homogeneous → existing GCN/GAT/SAGE/GIN/MLP modules (node- or graph-level)
    * heterogeneous → HeteroGraphRegressor (wraps a homo backbone via to_hetero)
"""
from __future__ import annotations

import logging
from typing import Optional

import pytorch_lightning as pl

log = logging.getLogger(__name__)

from app.models.gcn import GCNClassifier
from app.models.gat import GATClassifier
from app.models.sage import SAGEClassifier
from app.models.gin import GINClassifier
from app.models.mlp import MLPClassifier
from app.models.hetero_wrapper import HeteroGraphRegressor


HOMO_REGISTRY: dict[str, type[pl.LightningModule]] = {
    "gcn": GCNClassifier,
    "gat": GATClassifier,
    "sage": SAGEClassifier,
    "gin": GINClassifier,
    "mlp": MLPClassifier,
}

# Backbones to_hetero can lift. MLP/GIN are skipped (MLP has no edges; GIN's
# inner MLP doesn't play well with to_hetero's per-relation lift).
# GCNConv is excluded: it does NOT support bipartite message passing (src ≠ dst
# node type), which is the common case in heterogeneous graphs. GATConv and
# SAGEConv both handle bipartite edges correctly.
HETERO_BACKBONES = ("gat", "sage")


def get_model(
    model_name: str,
    num_features: int,
    num_classes: int = 2,
    task_type: str = "node_classification",
    metadata: Optional[tuple[list[str], list[tuple[str, str, str]]]] = None,
    **kwargs,
) -> pl.LightningModule:
    """Return a model instance.

    If ``metadata`` (HeteroData metadata tuple) is provided, returns a
    HeteroGraphRegressor wrapping the named backbone. Otherwise returns the
    standard homogeneous Lightning module.

    ``kwargs`` may include ``num_targets`` and ``loss_weights`` for multi-Y
    regression — both are passed through to the underlying module.
    """
    if metadata is not None:
        if model_name in HETERO_BACKBONES:
            conv = model_name
        else:
            conv = "sage"
            log.warning(
                "Model '%s' does not support bipartite message passing required for "
                "heterogeneous graphs; substituting 'sage' (SAGEConv) instead.",
                model_name,
            )
        return HeteroGraphRegressor(
            metadata=metadata,
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.3),
            lr=kwargs.get("lr", 1e-3),
            num_classes=num_classes,
            conv=conv,
            task_type=task_type,
            num_targets=kwargs.get("num_targets", 1),
            loss_weights=kwargs.get("loss_weights"),
        )

    if model_name not in HOMO_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(HOMO_REGISTRY)}")
    # class_weights may be passed through via kwargs for classification tasks.
    return HOMO_REGISTRY[model_name](
        num_features=num_features,
        num_classes=num_classes,
        task_type=task_type,
        **kwargs,
    )
