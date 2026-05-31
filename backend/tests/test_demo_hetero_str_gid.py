"""Tests for the v3 hetero demo: string Graph_IDs + multi-feature CAP groups.

Covers:
1. Excel parses with string Graph_IDs preserved as strings (no int coercion).
2. Each graph has the expected feature subset for its CAP group (A/B/C).
3. ``compute_per_graph_feature_schema`` reports per-graph variable subsets.
4. ``parsed_excel_to_hetero_list`` produces ``HeteroData`` with consistent
   tensor shapes across graphs (low-presence columns are excluded from the
   scaler but kept as zero-fill so dims line up).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.excel_ingestion import parse_excel_file
from app.data.feature_engineering import (
    compute_generic_explore,
    compute_per_graph_feature_schema,
)
from app.data.pyg_converter_hetero import parsed_excel_to_hetero_list


DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"
DEMO_FILE = DEMO_DIR / "demo_hetero_multifeature.v3.xlsx"


def _read_demo() -> bytes:
    if not DEMO_FILE.exists():
        pytest.skip(
            "demo_hetero_multifeature.v3.xlsx not generated; "
            "run scripts/generate_excel_demos.py"
        )
    return DEMO_FILE.read_bytes()


def test_v3_demo_preserves_string_graph_ids():
    parsed = parse_excel_file(_read_demo(), "hetero_v3")
    assert parsed["is_heterogeneous"] is True
    assert parsed["task_type"] == "graph_regression"
    assert parsed["label_column"] == "target_y"

    # Graph_IDs must remain strings (not int-coerced).
    g_ids = parsed["graph_df"]["_graph"].tolist()
    assert all(isinstance(g, str) for g in g_ids), \
        f"Expected string Graph_IDs, got types: {set(type(g) for g in g_ids)}"
    assert g_ids[0] == "G001"
    assert g_ids[-1] == f"G{len(g_ids):03d}"

    # Node sheet's _graph also uses strings (consistent with graph_df).
    node_g_ids = parsed["nodes_df"]["_graph"].tolist()
    assert all(isinstance(g, str) for g in node_g_ids)


def test_v3_demo_node_types():
    parsed = parse_excel_file(_read_demo(), "hetero_v3")
    spec = parsed["spec"]
    assert spec.node_types() == ["CAP", "RES"]
    assert spec.edge_types() == ["cap_res"]
    assert set(spec.x_columns("Node", "CAP")) == {"X_1", "X_2", "X_3"}
    assert set(spec.x_columns("Node", "RES")) == {"R_1", "R_2"}


def test_v3_demo_per_graph_feature_groups_vary():
    """G001=group A (X_1+X_2), G002=group B (X_1+X_3), G003=group C (X_2+X_3)."""
    parsed = parse_excel_file(_read_demo(), "hetero_v3")
    schema = compute_per_graph_feature_schema(parsed["node_dfs"])

    cap_graphs = schema["CAP"]["graphs"]
    # Different graphs must have different CAP feature subsets.
    assert set(cap_graphs["G001"]) == {"X_1", "X_2"}, \
        f"G001 expected X_1+X_2, got {cap_graphs['G001']}"
    assert set(cap_graphs["G002"]) == {"X_1", "X_3"}, \
        f"G002 expected X_1+X_3, got {cap_graphs['G002']}"
    assert set(cap_graphs["G003"]) == {"X_2", "X_3"}, \
        f"G003 expected X_2+X_3, got {cap_graphs['G003']}"

    # Union should cover all three CAP X columns; intersection should be empty.
    assert set(schema["CAP"]["union"]) == {"X_1", "X_2", "X_3"}
    assert schema["CAP"]["intersection"] == [], \
        "CAP intersection must be empty since groups carry disjoint pairs"

    # Each CAP X column appears in 2 of the 3 groups (e.g. X_1 in A and B),
    # so its graph-level presence ratio should be ~2/3.
    for col in ("X_1", "X_2", "X_3"):
        ratio = schema["CAP"]["presence_per_column"][col]
        assert 0.6 <= ratio <= 0.75, \
            f"CAP {col} presence ratio {ratio} outside expected [0.6, 0.75]"

    # RES has the same two features in every graph.
    assert set(schema["RES"]["union"]) == {"R_1", "R_2"}
    assert set(schema["RES"]["intersection"]) == {"R_1", "R_2"}


def test_v3_demo_explore_stats_round_trip():
    """compute_generic_explore must include presence_pct and the per-graph schema."""
    parsed = parse_excel_file(_read_demo(), "hetero_v3")
    spec = parsed["spec"]
    ntf = {t: spec.x_columns("Node", t) for t in spec.node_types()}
    etf = {t: spec.x_columns("Edge", t) for t in spec.edge_types()}

    stats = compute_generic_explore(
        parsed["nodes_df"], parsed["edges_df"],
        is_heterogeneous=True,
        node_types=spec.node_types(), edge_types=spec.edge_types(),
        canonical_edges=parsed["canonical_edges"],
        node_dfs=parsed["node_dfs"], edge_dfs=parsed["edge_dfs"],
        node_type_features=ntf, edge_type_features=etf,
    )
    assert stats["graph_count"] == 30
    assert stats["is_heterogeneous"] is True
    assert "per_graph_feature_schema" in stats
    assert "CAP" in stats["per_graph_feature_schema"]

    # Per-graph aware semantics: presence_pct is computed only over rows in
    # graphs that USE the column. Each X column is used in 2 of the 3 groups
    # (~20 graphs out of 30), and within those 20 graphs every CAP row has a
    # value for that X column → presence_pct should be 100% in-scope.
    # graph_presence_pct exposes the alternative "fraction of graphs using
    # this column" view (~66.7%).
    cap_cols = {c["name"]: c for c in stats["columns"] if c.get("node_type") == "CAP"}
    for col_name in ("X_1", "X_2", "X_3"):
        assert col_name in cap_cols
        assert "presence_pct" in cap_cols[col_name]
        assert "graph_presence_pct" in cap_cols[col_name]
        assert cap_cols[col_name]["presence_pct"] == 100.0, (
            f"{col_name} should be 100% in-scope, got "
            f"{cap_cols[col_name]['presence_pct']}"
        )
        # Each X column is used in 2 of the 3 feature groups (~20/30 graphs).
        assert 60.0 <= cap_cols[col_name]["graph_presence_pct"] <= 70.0


def test_v3_demo_converts_to_hetero_data_list():
    """parsed_excel_to_hetero_list must produce HeteroData with stable shapes."""
    parsed = parse_excel_file(_read_demo(), "hetero_v3")
    data_list, scalers, feature_cols, canonical_edges, excluded_cols = (
        parsed_excel_to_hetero_list({
            "node_dfs": parsed["node_dfs"],
            "edge_dfs": parsed["edge_dfs"],
            "graph_df": parsed["graph_df"],
            "label_column": parsed["label_column"],
            "canonical_edges": parsed["canonical_edges"],
            "task_type": parsed["task_type"],
        })
    )
    assert len(data_list) == 30

    # All graphs must share the same per-type feature dim (so DataLoader
    # collation works). Even when a graph's CAP is missing X_3, fillna(0.0)
    # keeps the column; only scaling is skipped.
    cap_feat_dim = data_list[0]["CAP"].x.shape[1]
    res_feat_dim = data_list[0]["RES"].x.shape[1]
    for d in data_list:
        assert d["CAP"].x.shape[1] == cap_feat_dim
        assert d["RES"].x.shape[1] == res_feat_dim

    # Each graph carries a scalar y (graph_regression).
    for d in data_list:
        assert d.y.shape == (1,)

    # Reverse edges are appended by ToUndirected.
    et = {tuple(t) for t in data_list[0].edge_types}
    assert ("CAP", "cap_res", "RES") in et
    assert ("RES", "rev_cap_res", "CAP") in et


def test_v3_demo_target_predictability():
    """Y values must vary in a sane range — not constant, not exploding.

    The generator builds Y from a small linear combination + tiny Gaussian
    noise, so the spread should be moderate and finite.
    """
    parsed = parse_excel_file(_read_demo(), "hetero_v3")
    y = parsed["graph_df"]["target_y"].astype(float)
    assert y.std() > 0.05, f"Target y has too little variation: std={y.std()}"
    assert y.std() < 5.0, f"Target y is unreasonably spread: std={y.std()}"
    assert y.between(0.0, 20.0).all(), \
        f"Target y outside expected band [0, 20]: range=[{y.min()}, {y.max()}]"
