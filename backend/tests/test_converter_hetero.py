"""Tests for pyg_converter_hetero — Step 3: min_presence_ratio scaler filtering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.pyg_converter_hetero import _fit_scalers, _build_single_hetero


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_node_dfs(n_rows: int = 20) -> dict[str, pd.DataFrame]:
    """Build a minimal node_dfs with one type 'A'.

    Column 'full' has all rows present.
    Column 'sparse' has only 5% of rows present (1 out of 20) — below 0.1 threshold.
    """
    rng = np.random.default_rng(42)
    data = {
        "node_id": list(range(n_rows)),
        "_graph": [1] * n_rows,
        "_node_type": ["A"] * n_rows,
        "full": rng.random(n_rows).tolist(),
        "sparse": [float(i) if i == 0 else None for i in range(n_rows)],
    }
    return {"A": pd.DataFrame(data)}


EXCLUDE = {"node_id", "_graph", "_node_type", "_edge_type",
           "src_id", "dst_id", "src_type", "dst_type", "Graph_ID",
           "Type", "Edge_Type"}


# ── test_scaler_excludes_low_presence ─────────────────────────────────────────


def test_scaler_excludes_low_presence():
    """Column 'sparse' with 5% presence (< 0.1) must be in excluded_cols, not in scaler."""
    node_dfs = _make_node_dfs(20)
    scalers, feature_cols, excluded_cols = _fit_scalers(node_dfs, EXCLUDE, min_presence_ratio=0.1)

    assert "A" in excluded_cols
    assert "sparse" in excluded_cols["A"], \
        f"'sparse' should be excluded, got: {excluded_cols['A']}"
    assert "full" not in excluded_cols["A"], \
        f"'full' should NOT be excluded, got: {excluded_cols['A']}"

    # scaler for A was fit only on 'full' column, so feature_names_in_ has 1 feature
    sc = scalers["A"]
    assert hasattr(sc, "mean_"), "Scaler should be fitted"
    assert sc.mean_.shape == (1,), \
        f"Scaler should have 1 feature (full only), got shape {sc.mean_.shape}"


# ── test_scaler_excluded_cols_persisted_in_artifact ───────────────────────────


def test_scaler_excluded_cols_persisted_in_artifact(tmp_path):
    """excluded_cols can be serialized alongside scalers and restored."""
    import torch

    node_dfs = _make_node_dfs(20)
    scalers, feature_cols, excluded_cols = _fit_scalers(node_dfs, EXCLUDE, min_presence_ratio=0.1)

    artifact = {
        "scalers": scalers,
        "feature_cols": feature_cols,
        "excluded_cols": excluded_cols,
    }
    path = tmp_path / "artifact.pt"
    torch.save(artifact, path)

    loaded = torch.load(path, weights_only=False)
    assert loaded["excluded_cols"] == excluded_cols
    assert "sparse" in loaded["excluded_cols"]["A"]


# ── test_scaler_skips_low_presence_columns ────────────────────────────────────


def test_scaler_skips_low_presence_columns():
    """Excluded 'sparse' column should appear as 0.0 in output tensor (fillna, no scaling)."""
    from torch_geometric.data import HeteroData

    node_dfs = _make_node_dfs(20)
    scalers, feature_cols, excluded_cols = _fit_scalers(node_dfs, EXCLUDE, min_presence_ratio=0.1)

    # Build a single hetero graph (graph_id=1, no edges, no graph_df)
    data = _build_single_hetero(
        graph_id=1,
        node_dfs=node_dfs,
        edge_dfs={},
        graph_df=None,
        label_columns=["label"],   # not used since graph_df=None
        canonical_edges=[],
        scalers=scalers,
        feature_cols=feature_cols,
        excluded_cols=excluded_cols,
    )

    x = data["A"].x.numpy()
    cols = feature_cols["A"]
    sparse_idx = cols.index("sparse")

    # sparse column: all rows except row 0 were NaN → fillna(0.0) → all 0.0
    # row 0 also became 0.0 via fillna (its original value was 0.0 too)
    assert (x[:, sparse_idx] == 0.0).all(), \
        f"Excluded col 'sparse' should be 0.0 everywhere, got: {x[:, sparse_idx]}"

    # 'full' column should have been scaled (mean≈0, std≈1 after StandardScaler)
    full_idx = cols.index("full")
    full_vals = x[:, full_idx]
    assert not (full_vals == 0.0).all(), \
        "Column 'full' should be scaled (non-trivially), not all zeros"
