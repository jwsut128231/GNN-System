"""Pipeline-level smoke tests for multi-Y training.

These tests construct a minimal in-memory dataset record (the same shape that
`projects.py::_store_excel_dataset` would build) and exercise the prepare
functions plus a tiny HPO/training pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from app.training.pipeline import (
    _prepare_graph_homo,
    _prepare_node,
    _label_columns_for,
)
from app.training.target_scaler import TargetScaler


def _make_multi_y_graph_dataset() -> dict:
    """Tiny multi-graph dataset with 6 graphs and two Y columns."""
    rng = np.random.default_rng(0)
    n_graphs = 6
    nodes = []
    edges = []
    graph_rows = []
    for gid in range(1, n_graphs + 1):
        for nid in range(4):
            nodes.append({
                "_graph": gid, "node_id": nid,
                "X_1": float(rng.uniform(0, 1)),
                "X_2": float(rng.uniform(0, 1)),
            })
        for s, d in [(0, 1), (1, 2), (2, 3)]:
            edges.append({"_graph": gid, "src_id": s, "dst_id": d})
        graph_rows.append({
            "_graph": gid,
            "y1": float(rng.uniform(0, 10)),
            "y2": float(rng.uniform(50, 100)),
        })
    return {
        "nodes_df": pd.DataFrame(nodes),
        "edges_df": pd.DataFrame(edges),
        "graph_df": pd.DataFrame(graph_rows),
        "label_column": "y1",
        "label_columns": ["y1", "y2"],
        "label_weights": [2.0, 0.5],
        "task_type": "graph_regression",
    }


# ── prepare returns multi-Y-shaped y tensors ──────────────────────────

def test_prepare_graph_homo_multi_y_shapes():
    dataset = _make_multi_y_graph_dataset()
    gen = torch.Generator().manual_seed(42)
    train, val, test, num_classes, scaler = _prepare_graph_homo(dataset, gen)
    assert num_classes == 1
    assert len(train) + len(val) + len(test) == 6
    # Each train Data.y is (1, T) after scaling — shape preserved.
    for d in train + val:
        assert d.y.shape == (1, 2), f"per-graph multi-Y must be (1, T), got {tuple(d.y.shape)}"
        assert d.num_targets == 2
    # Test items are NOT scaled — but shape is still (1, T).
    for d in test:
        assert d.y.shape == (1, 2)
    # Scaler is vector-valued.
    assert isinstance(scaler.mean, np.ndarray)
    assert scaler.mean.shape == (2,)


def test_prepare_node_multi_y_shapes():
    nodes_df = pd.DataFrame({
        "node_id": list(range(10)),
        "X_1": np.arange(10).astype(float) / 10,
        "y1": np.arange(10).astype(float),
        "y2": np.arange(10).astype(float) * 5,
    })
    edges_df = pd.DataFrame({"src_id": [0, 1, 2], "dst_id": [1, 2, 3]})
    dataset = {
        "nodes_df_train": nodes_df.iloc[:7].reset_index(drop=True),
        "nodes_df_test": nodes_df.iloc[7:].reset_index(drop=True),
        "edges_df_train": edges_df,
        "edges_df_test": pd.DataFrame({"src_id": [], "dst_id": []}),
        "label_column": "y1",
        "label_columns": ["y1", "y2"],
        "task_type": "node_regression",
    }
    train_data, test_data, _, scaler = _prepare_node(dataset)
    # train y is scaled, shape (N_train, T)
    assert train_data.y.shape == (7, 2)
    assert test_data.y.shape == (3, 2)
    assert isinstance(scaler.mean, np.ndarray)
    assert scaler.mean.shape == (2,)


# ── label_columns_for handles legacy single-Y datasets ────────────────

def test_label_columns_for_legacy_singular_only():
    assert _label_columns_for({"label_column": "y"}) == ["y"]


def test_label_columns_for_multi_y_record():
    assert _label_columns_for({
        "label_column": "y1",
        "label_columns": ["y1", "y2", "y3"],
    }) == ["y1", "y2", "y3"]


# ── End-to-end smoke: prepare → forward pass works ────────────────────

def test_multi_y_forward_pass_through_model():
    """After prepare, a model built with matching num_targets runs end-to-end."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from app.models.factory import get_model

    dataset = _make_multi_y_graph_dataset()
    gen = torch.Generator().manual_seed(0)
    train, val, _test, _nc, _scaler = _prepare_graph_homo(dataset, gen)
    loader = PyGDataLoader(train, batch_size=2, shuffle=False)
    model = get_model(
        "sage", num_features=2, num_classes=1, task_type="graph_regression",
        hidden_dim=8, num_layers=2, dropout=0.0, lr=1e-3,
        num_targets=2, loss_weights=torch.tensor([2.0, 0.5]),
    )
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, None, batch=batch.batch)
    # B (graphs in batch), T targets
    assert out.shape == (2, 2)
    # Batched y is (B, T)
    assert batch.y.shape == (2, 2)
    # Loss compiles without shape errors
    loss = model._shared_step(batch, "val")
    assert torch.isfinite(loss)
