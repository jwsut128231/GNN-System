"""Tests for multi-target model output + weighted loss."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.factory import get_model
from app.models.loss import weighted_regression_loss


# ── weighted_regression_loss ──────────────────────────────────────────

def test_weighted_regression_loss_single_target():
    out = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 2.0])
    loss = weighted_regression_loss(out, y, loss_weights=None, num_targets=1)
    # MSE = mean([0, 0, 1]) = 1/3
    assert abs(loss.item() - 1.0 / 3.0) < 1e-5


def test_weighted_regression_loss_multi_target_unweighted():
    out = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    y = torch.tensor([[1.0, 11.0], [3.0, 18.0]])
    # SE: [[0, 1], [1, 4]] → sum-per-sample [1, 5] → mean = 3.0
    loss = weighted_regression_loss(out, y, loss_weights=None, num_targets=2)
    assert abs(loss.item() - 3.0) < 1e-5


def test_weighted_regression_loss_multi_target_weighted():
    out = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    y = torch.tensor([[1.0, 11.0], [3.0, 18.0]])
    # weights [2, 0.5]: weighted SE per element [[0*2, 1*0.5], [1*2, 4*0.5]]
    # → sum-per-sample [0.5, 4.0] → mean = 2.25
    w = torch.tensor([2.0, 0.5])
    loss = weighted_regression_loss(out, y, loss_weights=w, num_targets=2)
    assert abs(loss.item() - 2.25) < 1e-5


# ── homogeneous model output shape ────────────────────────────────────

def test_homo_model_single_target_output_shape():
    """GCN regressor with num_targets=1 outputs [N] (legacy shape)."""
    model = get_model(
        "gcn", num_features=4, num_classes=1, task_type="node_regression",
        hidden_dim=8, num_layers=2, dropout=0.0, lr=1e-3, num_targets=1,
    )
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, None, batch=None)
    assert out.shape == (5,), f"single-Y node output must be (N,), got {tuple(out.shape)}"


def test_homo_model_multi_target_output_shape():
    """GCN regressor with num_targets=3 outputs [N, 3]."""
    model = get_model(
        "gcn", num_features=4, num_classes=1, task_type="node_regression",
        hidden_dim=8, num_layers=2, dropout=0.0, lr=1e-3, num_targets=3,
        loss_weights=torch.tensor([1.0, 2.0, 0.5]),
    )
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, None, batch=None)
    assert out.shape == (5, 3), f"multi-Y node output must be (N, T), got {tuple(out.shape)}"
    assert model.loss_weights.shape == (3,)


def test_homo_model_multi_target_graph_level():
    """SAGE regressor at graph level with num_targets=2 outputs [B, 2]."""
    model = get_model(
        "sage", num_features=4, num_classes=1, task_type="graph_regression",
        hidden_dim=8, num_layers=2, dropout=0.0, lr=1e-3, num_targets=2,
    )
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])  # two graphs
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, None, batch=batch)
    assert out.shape == (2, 2), f"multi-Y graph output must be (B, T), got {tuple(out.shape)}"


def test_homo_model_classification_unchanged_when_single_y():
    """Single-Y classification path is unchanged: output shape (N, num_classes)."""
    model = get_model(
        "gcn", num_features=4, num_classes=3, task_type="node_classification",
        hidden_dim=8, num_layers=2, dropout=0.0, lr=1e-3, num_targets=1,
    )
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, None, batch=None)
    assert out.shape == (5, 3)
