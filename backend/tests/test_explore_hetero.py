"""Heterogeneous explore-stats regression test.

Catches the "feature missing" bug where cross-type NaN padding in the unified
``nodes_df`` caused `compute_generic_explore` to report 60%+ missing for every
type-specific feature.
"""
from __future__ import annotations

import pandas as pd

from app.data.feature_engineering import compute_generic_explore, compute_per_graph_feature_schema


def test_hetero_per_type_explore_has_no_false_missing():
    # Two node types with entirely disjoint feature sets.
    cell_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "_graph": [1, 1, 1],
        "_node_type": ["cell"] * 3,
        "cell_area": [1.1, 2.2, 3.3],
        "cell_drive": [1, 2, 4],
    })
    pin_df = pd.DataFrame({
        "node_id": [3, 4],
        "_graph": [1, 1],
        "_node_type": ["pin"] * 2,
        "pin_cap": [0.5, 0.6],
    })
    unified = pd.concat([cell_df, pin_df], ignore_index=True)

    stats = compute_generic_explore(
        unified, pd.DataFrame(columns=["src_id", "dst_id"]),
        is_heterogeneous=True,
        node_types=["cell", "pin"], edge_types=[],
        canonical_edges=[],
        node_dfs={"cell": cell_df, "pin": pin_df},
        edge_dfs={},
    )
    by_name = {(c["name"], c.get("node_type")): c for c in stats["columns"]}
    # cell features should have 0 missing under cell type
    assert by_name[("cell_area", "cell")]["missing_count"] == 0
    assert by_name[("cell_drive", "cell")]["missing_count"] == 0
    # pin feature should have 0 missing under pin type
    assert by_name[("pin_cap", "pin")]["missing_count"] == 0
    # Crucially, cell_area should NOT appear as a column under pin (it was
    # dropped by ingestion) — and vice versa.
    assert ("cell_area", "pin") not in by_name
    assert ("pin_cap", "cell") not in by_name


def test_homogeneous_explore_unchanged():
    df = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "x1": [0.1, 0.2, 0.3, 0.4],
        "x2": [1.0, None, 3.0, 4.0],
    })
    stats = compute_generic_explore(df, pd.DataFrame(columns=["src_id", "dst_id"]))
    by_name = {c["name"]: c for c in stats["columns"]}
    # Legacy path uses unified counts: x2 has 1/4 missing.
    assert by_name["x2"]["missing_count"] == 1
    assert by_name["x1"]["missing_count"] == 0


def test_shared_feature_across_types_no_cross_contamination():
    """Shared feature 'area' declared for both cell and pin.

    The per-type DataFrames produced by _split_by_type keep ALL columns from
    the unified sheet — so 'f_pin' appears in cell's frame as all-NaN, and
    'f_cell' appears in pin's frame as all-NaN.  With declared_cols filtering
    (Fix 2), only the columns declared in the Parameter schema for each type
    should appear in that type's column stats.

    Assertions:
    - cell stats include 'area' and 'f_cell', but NOT 'f_pin'
    - pin stats include 'area' and 'f_pin', but NOT 'f_cell'
    - 'area' missing_count is 0 for both cell and pin (all values populated)
    """
    # Simulate unified DataFrame after _split_by_type — all columns present,
    # cross-type padding is NaN.
    cell_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "_graph": [1, 1, 1, 1],
        "_node_type": ["cell"] * 4,
        "area":   [1.0, 2.0, 3.0, 4.0],   # shared
        "f_cell": [10.0, 20.0, 30.0, 40.0],  # cell-only
        "f_pin":  [None, None, None, None],   # cross-type NaN padding
    })
    pin_df = pd.DataFrame({
        "node_id": [4, 5, 6, 7],
        "_graph": [1, 1, 1, 1],
        "_node_type": ["pin"] * 4,
        "area":   [0.5, 0.6, 0.7, 0.8],  # shared
        "f_pin":  [1.1, 2.2, 3.3, 4.4],  # pin-only
        "f_cell": [None, None, None, None],  # cross-type NaN padding
    })
    unified = pd.concat([cell_df, pin_df], ignore_index=True)

    node_type_features = {
        "cell": ["area", "f_cell"],
        "pin":  ["area", "f_pin"],
    }

    stats = compute_generic_explore(
        unified, pd.DataFrame(columns=["src_id", "dst_id"]),
        is_heterogeneous=True,
        node_types=["cell", "pin"], edge_types=[],
        canonical_edges=[],
        node_dfs={"cell": cell_df, "pin": pin_df},
        edge_dfs={},
        node_type_features=node_type_features,
    )
    by_key = {(c["name"], c.get("node_type")): c for c in stats["columns"]}

    # Shared feature has correct missing count for each type
    assert ("area", "cell") in by_key
    assert by_key[("area", "cell")]["missing_count"] == 0
    assert by_key[("area", "cell")]["missing_pct"] == 0.0

    assert ("area", "pin") in by_key
    assert by_key[("area", "pin")]["missing_count"] == 0

    # Type-private features present under their own type
    assert ("f_cell", "cell") in by_key
    assert by_key[("f_cell", "cell")]["missing_count"] == 0
    assert ("f_pin", "pin") in by_key
    assert by_key[("f_pin", "pin")]["missing_count"] == 0

    # Cross-type padding columns must NOT appear
    assert ("f_pin", "cell") not in by_key, "f_pin must not appear in cell stats"
    assert ("f_cell", "pin") not in by_key, "f_cell must not appear in pin stats"


def test_missing_pct_uses_type_row_count_as_denominator():
    """cell's area has 2/4 rows missing → missing% must be 50%, NOT 25%.

    Without Fix 2, if missing% used the full 8-row unified frame as denominator
    it would incorrectly show 25%.
    """
    cell_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "_graph": [1, 1, 1, 1],
        "_node_type": ["cell"] * 4,
        "area": [1.0, None, 3.0, None],  # 2/4 missing in cell rows
        "f_pin": [None, None, None, None],  # cross-type padding
    })
    pin_df = pd.DataFrame({
        "node_id": [4, 5, 6, 7],
        "_graph": [1, 1, 1, 1],
        "_node_type": ["pin"] * 4,
        "area":  [0.5, 0.6, 0.7, 0.8],
        "f_pin": [1.1, 2.2, 3.3, 4.4],
    })
    unified = pd.concat([cell_df, pin_df], ignore_index=True)

    node_type_features = {
        "cell": ["area"],
        "pin":  ["area", "f_pin"],
    }

    stats = compute_generic_explore(
        unified, pd.DataFrame(columns=["src_id", "dst_id"]),
        is_heterogeneous=True,
        node_types=["cell", "pin"], edge_types=[],
        canonical_edges=[],
        node_dfs={"cell": cell_df, "pin": pin_df},
        edge_dfs={},
        node_type_features=node_type_features,
    )
    by_key = {(c["name"], c.get("node_type")): c for c in stats["columns"]}

    cell_area = by_key[("area", "cell")]
    # Denominator must be 4 (cell rows only), not 8 (whole frame)
    assert cell_area["missing_count"] == 2
    assert cell_area["missing_pct"] == 50.0, (
        f"Expected 50.0% missing for cell area, got {cell_area['missing_pct']}% "
        "(denominator should be 4 cell rows, not 8 total rows)"
    )

    # pin's area has 0 missing
    assert by_key[("area", "pin")]["missing_count"] == 0
    assert by_key[("area", "pin")]["missing_pct"] == 0.0


def test_hetero_three_types_cell_area_not_missing():
    """Simulate real chip-design hetero graph with cell/pin/net node types.

    The unified DataFrame has NaN cross-contamination (cell_area is NaN for
    pin and net rows, pin_cap is NaN for cell and net rows, etc.).  When
    per-type DataFrames are supplied, missing_count for each feature must be 0
    because each type only carries its own columns.
    """
    cell_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "_graph": [1, 1, 1],
        "_node_type": ["cell"] * 3,
        "cell_area": [1.1, 2.2, 3.3],
        "cell_drive": [1, 2, 4],
    })
    pin_df = pd.DataFrame({
        "node_id": [3, 4, 5, 6],
        "_graph": [1, 1, 1, 1],
        "_node_type": ["pin"] * 4,
        "pin_cap": [0.5, 0.6, 0.7, 0.8],
    })
    net_df = pd.DataFrame({
        "node_id": [7, 8],
        "_graph": [1, 1],
        "_node_type": ["net"] * 2,
        "net_fanout": [3.0, 5.0],
    })
    # The unified concat mimics what parse_excel_file returns for nodes_df.
    # cell_area, cell_drive, pin_cap, net_fanout are NaN for unrelated types.
    unified = pd.concat([cell_df, pin_df, net_df], ignore_index=True)

    # Sanity-check: the unified frame DOES have NaN cross-contamination.
    assert unified["cell_area"].isna().sum() == 6  # pin (4) + net (2) rows
    assert unified["pin_cap"].isna().sum() == 5    # cell (3) + net (2) rows
    assert unified["net_fanout"].isna().sum() == 7  # cell (3) + pin (4) rows

    stats = compute_generic_explore(
        unified, pd.DataFrame(columns=["src_id", "dst_id"]),
        is_heterogeneous=True,
        node_types=["cell", "pin", "net"],
        edge_types=[],
        canonical_edges=[],
        node_dfs={"cell": cell_df, "pin": pin_df, "net": net_df},
        edge_dfs={},
    )

    by_name = {(c["name"], c.get("node_type")): c for c in stats["columns"]}

    # Each type-scoped feature must have 0 missing under its own type.
    assert by_name[("cell_area", "cell")]["missing_count"] == 0
    assert by_name[("cell_drive", "cell")]["missing_count"] == 0
    assert by_name[("pin_cap", "pin")]["missing_count"] == 0
    assert by_name[("net_fanout", "net")]["missing_count"] == 0

    # Cross-type columns must not appear (each per-type df only has its own cols).
    assert ("cell_area", "pin") not in by_name
    assert ("cell_area", "net") not in by_name
    assert ("pin_cap", "cell") not in by_name
    assert ("net_fanout", "cell") not in by_name


# ── Step 2: per-graph feature schema + presence rate tests ─────────────────

def test_per_graph_feature_schema():
    """CAP type: graph_1 has [A, B], graph_2 has [A, C].

    Expected: union=[A,B,C], intersection=[A],
    presence_per_column={A:1.0, B:0.5, C:0.5}
    """
    cap_df = pd.DataFrame({
        "_graph": ["graph_1", "graph_1", "graph_2", "graph_2"],
        "_node_type": ["CAP"] * 4,
        "A": [1.0, 2.0, 3.0, 4.0],   # present in both graphs
        "B": [0.1, 0.2, None, None],   # present only in graph_1
        "C": [None, None, 0.3, 0.4],   # present only in graph_2
    })

    schema = compute_per_graph_feature_schema({"CAP": cap_df})

    assert "CAP" in schema
    cap = schema["CAP"]

    assert sorted(cap["union"]) == ["A", "B", "C"]
    assert cap["intersection"] == ["A"]
    assert cap["presence_per_column"]["A"] == 1.0
    assert cap["presence_per_column"]["B"] == 0.5
    assert cap["presence_per_column"]["C"] == 0.5
    assert cap["low_presence_columns"] == []


def test_presence_pct_in_column_stats():
    """Column stats include presence_pct and low_presence_warning fields."""
    cell_df = pd.DataFrame({
        "_graph": [1, 1, 1, 1],
        "_node_type": ["cell"] * 4,
        "area": [1.0, 2.0, None, 4.0],  # 3/4 present → 75%
        "score": [None, None, None, None],  # 0/4 present → 0%
    })
    unified = cell_df.copy()

    stats = compute_generic_explore(
        unified, pd.DataFrame(columns=["src_id", "dst_id"]),
        is_heterogeneous=True,
        node_types=["cell"], edge_types=[],
        canonical_edges=[],
        node_dfs={"cell": cell_df},
        edge_dfs={},
    )
    by_key = {(c["name"], c.get("node_type")): c for c in stats["columns"]}

    area = by_key[("area", "cell")]
    assert "presence_pct" in area
    assert "low_presence_warning" in area
    assert area["presence_pct"] == 75.0

    score = by_key[("score", "cell")]
    assert score["presence_pct"] == 0.0


def test_low_presence_warning_flagged():
    """Column D present in 5% of graphs (1 of 20) → low_presence_warning True."""
    rows = []
    for gid in range(20):
        # Column D only has a value in graph_0
        d_val = 1.0 if gid == 0 else None
        rows.append({"_graph": gid, "_node_type": "T", "E": float(gid), "D": d_val})
    t_df = pd.DataFrame(rows)

    schema = compute_per_graph_feature_schema(
        {"T": t_df},
        min_presence_ratio=0.1,  # 10% threshold
    )

    t = schema["T"]
    # D is present in only 1/20 = 5% of graphs → below 10% threshold
    assert "D" in t["low_presence_columns"], (
        f"D should be low-presence. presence={t['presence_per_column'].get('D')}, "
        f"low_presence_columns={t['low_presence_columns']}"
    )
    # E is present in all 20 graphs → not flagged
    assert "E" not in t["low_presence_columns"]
