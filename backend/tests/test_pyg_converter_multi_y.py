"""Tests for multi-target (multi-Y) PyG converter behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from app.data.pyg_converter import (
    dataframes_to_pyg_dynamic,
    dataframes_to_graph_list,
)
from app.data.pyg_converter_hetero import parsed_excel_to_hetero_list


# ── Single-Y backwards compatibility ───────────────────────────────────

def test_single_y_node_regression_shape_unchanged():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "X_1": [0.1, 0.2, 0.3, 0.4],
        "y": [1.0, 2.0, 3.0, 4.0],
    })
    edges_df = pd.DataFrame({"src_id": [0, 1], "dst_id": [1, 2]})
    data, _, _ = dataframes_to_pyg_dynamic(
        nodes_df, edges_df, label_column="y", task_type="node_regression",
    )
    assert data.y.shape == (4,), f"single-Y y must stay 1-D, got {tuple(data.y.shape)}"
    assert data.num_targets == 1


def test_single_y_node_regression_list_arg_compat():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1], "X_1": [0.1, 0.2], "y": [1.0, 2.0],
    })
    data, _, _ = dataframes_to_pyg_dynamic(
        nodes_df, pd.DataFrame({"src_id": [], "dst_id": []}),
        label_column=["y"], task_type="node_regression",
    )
    assert data.y.shape == (2,)
    assert data.num_targets == 1


# ── Multi-Y node regression ────────────────────────────────────────────

def test_multi_y_node_regression_y_shape():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "X_1": [0.1, 0.2, 0.3, 0.4],
        "y1": [1.0, 2.0, 3.0, 4.0],
        "y2": [10.0, 20.0, 30.0, 40.0],
    })
    edges_df = pd.DataFrame({"src_id": [0, 1], "dst_id": [1, 2]})
    data, _, _ = dataframes_to_pyg_dynamic(
        nodes_df, edges_df,
        label_column=["y1", "y2"], task_type="node_regression",
    )
    assert data.y.shape == (4, 2), f"multi-Y y must be (N, T), got {tuple(data.y.shape)}"
    assert data.num_targets == 2
    # Values preserved
    assert torch.allclose(data.y[:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(data.y[:, 1], torch.tensor([10.0, 20.0, 30.0, 40.0]))


def test_multi_y_label_columns_excluded_from_features():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "X_1": [0.1, 0.2, 0.3],
        "y1": [1.0, 2.0, 3.0],
        "y2": [4.0, 5.0, 6.0],
    })
    data, _, feature_names = dataframes_to_pyg_dynamic(
        nodes_df, pd.DataFrame({"src_id": [], "dst_id": []}),
        label_column=["y1", "y2"], task_type="node_regression",
    )
    assert "y1" not in feature_names
    assert "y2" not in feature_names
    assert "X_1" in feature_names


# ── Multi-Y graph regression ───────────────────────────────────────────

def test_multi_y_graph_regression_per_graph_y_shape():
    nodes_df = pd.DataFrame({
        "_graph": [1, 1, 2, 2],
        "node_id": [0, 1, 0, 1],
        "X_1": [0.1, 0.2, 0.3, 0.4],
    })
    edges_df = pd.DataFrame({
        "_graph": [1, 2], "src_id": [0, 0], "dst_id": [1, 1],
    })
    graph_df = pd.DataFrame({
        "_graph": [1, 2],
        "y1": [10.0, 20.0],
        "y2": [100.0, 200.0],
    })
    data_list, _, _, num_classes = dataframes_to_graph_list(
        nodes_df, edges_df, graph_df,
        label_column=["y1", "y2"], task_type="graph_regression",
    )
    assert num_classes == 1
    assert len(data_list) == 2
    for d in data_list:
        # Shape (1, T) so PyG batching gives (B, T).
        assert d.y.shape == (1, 2), f"per-graph multi-Y y must be (1, T), got {tuple(d.y.shape)}"
        assert d.num_targets == 2
    # Values
    assert torch.allclose(data_list[0].y, torch.tensor([[10.0, 100.0]]))
    assert torch.allclose(data_list[1].y, torch.tensor([[20.0, 200.0]]))


def test_single_y_graph_regression_per_graph_y_shape_unchanged():
    nodes_df = pd.DataFrame({
        "_graph": [1, 2], "node_id": [0, 0], "X_1": [0.1, 0.2],
    })
    edges_df = pd.DataFrame({"_graph": [], "src_id": [], "dst_id": []})
    graph_df = pd.DataFrame({"_graph": [1, 2], "y": [10.0, 20.0]})
    data_list, _, _, _ = dataframes_to_graph_list(
        nodes_df, edges_df, graph_df,
        label_column="y", task_type="graph_regression",
    )
    for d in data_list:
        assert d.y.shape == (1,), f"single-Y per-graph y must be (1,), got {tuple(d.y.shape)}"
        assert d.num_targets == 1


# ── Multi-Y hetero ────────────────────────────────────────────────────

def test_multi_y_hetero_graph_regression():
    """parsed_excel_to_hetero_list emits y of shape (T,) per HeteroData."""
    parsed = {
        "node_dfs": {
            "cell": pd.DataFrame({
                "_graph": [1, 1, 2, 2], "node_id": [0, 1, 0, 1],
                "cell_area": [1.0, 2.0, 3.0, 4.0],
            }),
            "pin": pd.DataFrame({
                "_graph": [1, 1, 2, 2], "node_id": [10, 11, 10, 11],
                "pin_cap": [0.1, 0.2, 0.3, 0.4],
            }),
        },
        "edge_dfs": {
            "cell2pin": pd.DataFrame({
                "_graph": [1, 2], "src_id": [0, 0], "dst_id": [10, 10],
                "src_type": ["cell", "cell"], "dst_type": ["pin", "pin"],
            }),
        },
        "graph_df": pd.DataFrame({
            "_graph": [1, 2], "y1": [5.0, 7.0], "y2": [50.0, 70.0],
        }),
        "label_columns": ["y1", "y2"],
        "label_column": "y1",
        "canonical_edges": [("cell", "cell2pin", "pin")],
    }
    data_list, _scalers, _feat, _ce, _exc = parsed_excel_to_hetero_list(parsed)
    assert len(data_list) == 2
    for d in data_list:
        # Shape (1, T) for PyG batching.
        assert d.y.shape == (1, 2), f"hetero multi-Y y must be (1, T), got {tuple(d.y.shape)}"
        assert d.num_targets == 2
    assert torch.allclose(data_list[0].y, torch.tensor([[5.0, 50.0]]))


def test_single_y_hetero_backwards_compat():
    """Old callers that pass only label_column still produce (1,) y."""
    parsed = {
        "node_dfs": {
            "cell": pd.DataFrame({
                "_graph": [1, 2], "node_id": [0, 0], "cell_area": [1.0, 2.0],
            }),
            "pin": pd.DataFrame({
                "_graph": [1, 2], "node_id": [10, 10], "pin_cap": [0.1, 0.2],
            }),
        },
        "edge_dfs": {
            "cell2pin": pd.DataFrame({
                "_graph": [1, 2], "src_id": [0, 0], "dst_id": [10, 10],
                "src_type": ["cell", "cell"], "dst_type": ["pin", "pin"],
            }),
        },
        "graph_df": pd.DataFrame({"_graph": [1, 2], "y": [5.0, 7.0]}),
        "label_column": "y",          # legacy key only
        "canonical_edges": [("cell", "cell2pin", "pin")],
    }
    data_list, _, _, _, _ = parsed_excel_to_hetero_list(parsed)
    for d in data_list:
        assert d.y.shape == (1,)
        assert d.num_targets == 1
