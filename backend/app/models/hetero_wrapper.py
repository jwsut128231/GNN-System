"""Heterogeneous GNN wrapper for graph-level tasks.

Uses ``torch_geometric.nn.to_hetero`` to lift a homogeneous backbone (GAT /
SAGE) into a heterogeneous one, then pools per-type node embeddings and feeds a
linear head.

Conv choice note
----------------
Only ``SAGEConv`` and ``GATConv`` are supported as backbones.  ``GCNConv`` is
explicitly excluded because it does **not** support bipartite message passing
(i.e., edges where source and destination node types differ), which is the
typical case in heterogeneous graphs.  The factory (``app.models.factory``)
maps any unsupported conv choice to ``"sage"`` and emits a warning.

User-selected ``model_family`` is only honored in homogeneous mode.  For hetero
training the effective conv is always ``"gat"`` or ``"sage"``.

Scope: graph_regression / graph_classification. Node-level hetero prediction is
deferred.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import (
    GATConv, SAGEConv, global_mean_pool, to_hetero,
)
from app.models._lr import build_scheduler

from app.models.loss import weighted_regression_loss


class _HomoBackbone(nn.Module):
    """A plain, type-agnostic GNN body suitable for ``to_hetero`` lifting.

    We keep it minimal — a stack of convs with ReLU + dropout. ``to_hetero``
    will duplicate each layer per relation and aggregate across relations.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_layers: int,
                 dropout: float, conv: str):
        super().__init__()
        conv_cls = {
            "gat": GATConv,
            "sage": SAGEConv,
        }[conv]
        # to_hetero() duplicates each conv per relation. GATConv defaults to
        # ``add_self_loops=True`` which is invalid when an edge connects two
        # different node types (source != target). Force it off so the lift
        # works across cross-type relations. SAGEConv does not have this flag.
        extra: dict = {}
        if conv == "gat":
            extra["add_self_loops"] = False
        self.convs = nn.ModuleList()
        # First conv goes from -1 (lazy init so to_hetero can wire per-type dims)
        self.convs.append(conv_cls(-1, hidden_dim, **extra))
        for _ in range(num_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim, **extra))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HeteroGraphRegressor(pl.LightningModule):
    """to_hetero-wrapped GNN with per-type mean-pool + linear head.

    Works for ``graph_regression`` (scalar or vector) and
    ``graph_classification`` (logits). Multi-target regression is enabled by
    passing ``num_targets > 1`` and an optional ``loss_weights`` tensor of
    length T.
    """
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        lr: float = 1e-3,
        num_classes: int = 1,
        conv: str = "sage",
        task_type: str = "graph_regression",
        num_targets: int = 1,
        loss_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["metadata", "loss_weights"])
        self.lr = lr
        self.task_type = task_type
        self.metadata = metadata
        self.num_targets = int(num_targets)
        if loss_weights is not None:
            self.register_buffer(
                "loss_weights", torch.as_tensor(loss_weights, dtype=torch.float),
            )
        else:
            self.loss_weights = None

        backbone = _HomoBackbone(
            num_features=-1, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, conv=conv,
        )
        self.hetero_gnn = to_hetero(backbone, metadata, aggr="mean")
        self.node_types = list(metadata[0])
        # For regression: emit T outputs (num_classes==1).
        # For classification: multi-Y is rejected upstream so num_targets==1.
        self.head = nn.Linear(hidden_dim * len(self.node_types), num_classes * self.num_targets)

    # ── inference ──

    def forward(self, x_dict, edge_index_dict, batch_dict):
        h_dict = self.hetero_gnn(x_dict, edge_index_dict)
        pooled = []
        for nt in self.node_types:
            h = h_dict.get(nt)
            b = batch_dict.get(nt)
            if h is None or b is None or h.numel() == 0:
                # Empty type in this batch — contribute zeros of expected shape.
                batch_size = int(max((v.max().item() + 1) for v in batch_dict.values()
                                     if v is not None and v.numel() > 0) or 1)
                pooled.append(torch.zeros(
                    batch_size, self.hparams.hidden_dim, device=self.device,
                ))
            else:
                pooled.append(global_mean_pool(h, b))
        z = torch.cat(pooled, dim=-1)
        out = self.head(z)
        if self.task_type.endswith("regression") and self.num_targets == 1:
            out = out.squeeze(-1)
        return out

    # ── pytorch-lightning hooks ──

    def _shared_step(self, batch, stage: str):
        x_dict = {nt: batch[nt].x for nt in batch.node_types}
        edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
        batch_dict = {nt: batch[nt].batch for nt in batch.node_types}
        out = self(x_dict, edge_index_dict, batch_dict)

        if self.task_type.endswith("regression"):
            loss = weighted_regression_loss(out, batch.y, self.loss_weights, self.num_targets)
            mae = (out - batch.y).abs().mean()
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.y.size(0))
            self.log(f"{stage}_mae", mae, prog_bar=False, batch_size=batch.y.size(0))
        else:
            loss = F.cross_entropy(out, batch.y.long())
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = build_scheduler(opt)
        return {"optimizer": opt, "lr_scheduler": sched}
