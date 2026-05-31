import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, BatchNorm1d
from app.models._lr import build_scheduler

from app.models.loss import weighted_regression_loss


class MLPClassifier(pl.LightningModule):
    """MLP baseline that ignores graph structure. Uses only node features."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        lr: float = 1e-3,
        class_weights: torch.Tensor | None = None,
        task_type: str = "node_classification",
        num_targets: int = 1,
        loss_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "loss_weights"])
        self.lr = lr
        self.class_weights = class_weights
        self.task_type = task_type
        self.num_targets = int(num_targets)
        if loss_weights is not None:
            self.register_buffer(
                "loss_weights", torch.as_tensor(loss_weights, dtype=torch.float),
            )
        else:
            self.loss_weights = None

        layers = []
        layers.append(Linear(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim))
        self.layers = torch.nn.ModuleList(layers)
        self.bns = torch.nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = Linear(hidden_dim, num_classes * self.num_targets)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.task_type.startswith("graph"):
            x = global_mean_pool(x, batch)
        out = self.classifier(x)
        if self.task_type.endswith("regression") and self.num_targets == 1:
            out = out.squeeze(-1)
        return out

    def _shared_step(self, batch, stage: str):
        out = self(batch.x, batch.edge_index if hasattr(batch, "edge_index") else None, batch.edge_attr if hasattr(batch, "edge_attr") else None, batch=batch.batch if hasattr(batch, "batch") else None)
        if self.task_type.endswith("regression"):
            loss = weighted_regression_loss(out, batch.y, self.loss_weights, self.num_targets)
            mae = (out - batch.y).abs().mean()
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.num_nodes)
            self.log(f"{stage}_mae", mae, prog_bar=False, batch_size=batch.num_nodes)
        else:
            weight = self.class_weights.to(out.device) if self.class_weights is not None else None
            loss = F.cross_entropy(out, batch.y, weight=weight)
            preds = out.argmax(dim=-1)
            acc = (preds == batch.y).float().mean()
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.num_nodes)
            self.log(f"{stage}_acc", acc, prog_bar=True, batch_size=batch.num_nodes)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = build_scheduler(opt)
        return {"optimizer": opt, "lr_scheduler": sched}
