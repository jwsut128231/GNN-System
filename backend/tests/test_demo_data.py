"""Smoke tests for the bundled demo .xlsx files.

Verifies that both the homogeneous and heterogeneous demo workbooks under
``backend/demo_data/`` parse cleanly through ``parse_excel_file``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.excel_ingestion import parse_excel_file
from app.data.feature_engineering import compute_generic_explore

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def _read(name: str) -> bytes:
    p = DEMO_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not generated; run scripts/generate_excel_demos.py")
    return p.read_bytes()


def test_demo_homo_parses_and_is_homogeneous():
    parsed = parse_excel_file(_read("demo_multigraph_homo.v2.xlsx"), "homo")
    assert parsed["is_heterogeneous"] is False
    assert parsed["task_type"] == "graph_regression"
    assert parsed["label_column"] == "target_delay"
    # 30 graphs bundled.
    assert parsed["graph_df"] is not None
    assert len(parsed["graph_df"]) == 30


def _read_hetero() -> bytes:
    """Read the hetero demo, preferring the canonical v2 file.

    The ``.new.xlsx`` fallback exists only for when the canonical file is locked
    by Excel during generation.  We prefer ``.v2.xlsx`` first so tests always
    validate the freshly generated canonical copy.
    """
    for name in ("demo_multigraph_hetero.v2.xlsx", "demo_multigraph_hetero.v2.new.xlsx"):
        p = DEMO_DIR / name
        if p.exists():
            return p.read_bytes()
    pytest.skip("hetero demo not generated; run scripts/generate_excel_demos.py")


def test_demo_hetero_parses_and_is_heterogeneous():
    parsed = parse_excel_file(_read_hetero(), "hetero")
    assert parsed["is_heterogeneous"] is True
    assert parsed["task_type"] == "graph_regression"
    # At least 2 node types present
    assert len(parsed["node_dfs"]) >= 2
    # Unified view contains all rows from all types
    total_typed = sum(len(df) for df in parsed["node_dfs"].values())
    assert len(parsed["nodes_df"]) == total_typed
    assert parsed["graph_df"] is not None
    # canonical_edges must be exactly the 2 observed triples, not a Cartesian product
    triples = set(map(tuple, parsed["canonical_edges"]))
    assert triples == {("cell", "cell_pin", "pin"), ("pin", "pin_net", "net")}, \
        f"Unexpected canonical_edges: {triples}"


def _isolated_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, graph_id: int) -> list:
    """Return list of node IDs in graph_id that appear in no edge."""
    g_nodes = nodes_df[nodes_df["_graph"] == graph_id]["node_id"].tolist()
    if edges_df.empty:
        return g_nodes
    g_edges = edges_df[edges_df["_graph"] == graph_id]
    connected = set(g_edges["src_id"].tolist()) | set(g_edges["dst_id"].tolist())
    return [n for n in g_nodes if n not in connected]


def _count_components(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, graph_id: int) -> int:
    """Count connected components in graph_id using Union-Find over node_ids."""
    g_nodes = nodes_df[nodes_df["_graph"] == graph_id]["node_id"].tolist()
    if not g_nodes:
        return 0
    parent = {n: n for n in g_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    if not edges_df.empty:
        g_edges = edges_df[edges_df["_graph"] == graph_id]
        for _, row in g_edges.iterrows():
            src, dst = row["src_id"], row["dst_id"]
            if src in parent and dst in parent:
                union(src, dst)

    return len({find(n) for n in g_nodes})


def test_demo_homo_no_isolated_nodes_first_graph():
    """First graph of the homo demo must have zero isolated nodes."""
    parsed = parse_excel_file(_read("demo_multigraph_homo.v2.xlsx"), "homo")
    first_gid = parsed["nodes_df"]["_graph"].iloc[0]
    isolated = _isolated_nodes(parsed["nodes_df"], parsed["edges_df"], first_gid)
    assert isolated == [], f"Isolated nodes in graph {first_gid}: {isolated}"


def test_demo_hetero_no_isolated_nodes_first_graph():
    """First graph of the hetero demo must have zero isolated nodes."""
    parsed = parse_excel_file(_read_hetero(), "hetero")
    first_gid = parsed["nodes_df"]["_graph"].iloc[0]
    isolated = _isolated_nodes(parsed["nodes_df"], parsed["edges_df"], first_gid)
    assert isolated == [], f"Isolated nodes in graph {first_gid}: {isolated}"


def test_demo_homo_first_graph_connected():
    """First graph of the homo demo must be a single connected component."""
    parsed = parse_excel_file(_read("demo_multigraph_homo.v2.xlsx"), "homo")
    first_gid = parsed["nodes_df"]["_graph"].iloc[0]
    n_components = _count_components(parsed["nodes_df"], parsed["edges_df"], first_gid)
    assert n_components == 1, (
        f"Graph {first_gid} has {n_components} connected components (expected 1)"
    )


def test_demo_hetero_first_graph_connected():
    """First graph of the hetero demo must be a single connected component."""
    parsed = parse_excel_file(_read_hetero(), "hetero")
    first_gid = parsed["nodes_df"]["_graph"].iloc[0]
    n_components = _count_components(parsed["nodes_df"], parsed["edges_df"], first_gid)
    assert n_components == 1, (
        f"Graph {first_gid} has {n_components} connected components (expected 1)"
    )


def test_demo_hetero_has_shared_node_features():
    """The hetero demo must exercise shared features declared for multiple types.

    area_um2  → declared for cell AND pin
    cap_ff    → declared for pin  AND net
    """
    parsed = parse_excel_file(_read_hetero(), "hetero")
    spec = parsed["spec"]

    # Parameter sheet must declare area_um2 for both cell and pin
    cell_x = spec.x_columns("Node", "cell")
    pin_x = spec.x_columns("Node", "pin")
    net_x = spec.x_columns("Node", "net")

    assert "area_um2" in cell_x, f"area_um2 not declared for cell; cell_x={cell_x}"
    assert "area_um2" in pin_x, f"area_um2 not declared for pin; pin_x={pin_x}"
    assert "cap_ff" in pin_x, f"cap_ff not declared for pin; pin_x={pin_x}"
    assert "cap_ff" in net_x, f"cap_ff not declared for net; net_x={net_x}"

    # Per-type frames must have no missing values for their declared features
    cell_df = parsed["node_dfs"]["cell"]
    pin_df = parsed["node_dfs"]["pin"]
    net_df = parsed["node_dfs"]["net"]

    assert cell_df["area_um2"].isna().sum() == 0, "cell area_um2 has unexpected NaN"
    assert pin_df["area_um2"].isna().sum() == 0, "pin area_um2 has unexpected NaN"
    assert pin_df["cap_ff"].isna().sum() == 0, "pin cap_ff has unexpected NaN"
    assert net_df["cap_ff"].isna().sum() == 0, "net cap_ff has unexpected NaN"

    # compute_generic_explore with declared feature lists must NOT include
    # cap_ff/net_fanout/pin_direction in cell's column stats (Fix 2 verification).
    node_type_features = {t: spec.x_columns("Node", t) for t in spec.node_types()}
    edge_type_features = {t: spec.x_columns("Edge", t) for t in spec.edge_types()}
    stats = compute_generic_explore(
        parsed["nodes_df"], parsed["edges_df"],
        is_heterogeneous=True,
        node_types=spec.node_types(),
        edge_types=spec.edge_types(),
        canonical_edges=parsed["canonical_edges"],
        node_dfs=parsed["node_dfs"],
        edge_dfs=parsed["edge_dfs"],
        node_type_features=node_type_features,
        edge_type_features=edge_type_features,
    )
    by_key = {(c["name"], c.get("node_type")): c for c in stats["columns"]}

    # Declared shared features present under correct types with 0 missing
    assert ("area_um2", "cell") in by_key, "area_um2 missing from cell stats"
    assert by_key[("area_um2", "cell")]["missing_count"] == 0
    assert ("area_um2", "pin") in by_key, "area_um2 missing from pin stats"
    assert by_key[("area_um2", "pin")]["missing_count"] == 0
    assert ("cap_ff", "pin") in by_key, "cap_ff missing from pin stats"
    assert by_key[("cap_ff", "pin")]["missing_count"] == 0
    assert ("cap_ff", "net") in by_key, "cap_ff missing from net stats"
    assert by_key[("cap_ff", "net")]["missing_count"] == 0

    # Non-declared columns must NOT appear under the wrong type
    assert ("cap_ff", "cell") not in by_key, "cap_ff wrongly appears in cell stats"
    assert ("net_fanout", "cell") not in by_key, "net_fanout wrongly appears in cell stats"
    assert ("pin_direction", "cell") not in by_key, "pin_direction wrongly appears in cell stats"
    assert ("area_um2", "net") not in by_key, "area_um2 wrongly appears in net stats"
