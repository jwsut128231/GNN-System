"""Tests for Excel template ingestion (V3 schema — 2026-04-26).

Schema: one sheet per level (Node / Edge / Graph).
Data sheets MAY have a Type column:
    - Absent or single-valued → homogeneous (single key "default").
    - Multi-valued → heterogeneous; rows split into per-type DataFrames.
Parameter sheet may declare multiple Type values per Level for hetero graphs.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from app.data.excel_ingestion import parse_excel_file
from app.data.excel_spec import parse_parameter_sheet, validate_hetero_consistency


def _build_workbook(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Materialise a dict of DataFrames as .xlsx bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return buf.read()


def _node_y_classification_workbook() -> bytes:
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_2", "Weight": None},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label", "Weight": 2.0},
        {"XY": "X", "Level": "Edge", "Type": "default", "Parameter": "E_1", "Weight": None},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1] * 5,
        "Node": [0, 1, 2, 3, 4],
        "X_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "X_2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "label": [0, 1, 0, 1, 0],
    })
    edges = pd.DataFrame({
        "Graph_ID": [1, 1],
        "Source_Node_ID": [0, 1],
        "Target_Node_ID": [1, 2],
        "E_1": [0.5, 0.7],
    })
    return _build_workbook({
        "Parameter": parameter,
        "Node": nodes,
        "Edge": edges,
    })


def _graph_y_regression_workbook() -> bytes:
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "X", "Level": "Graph", "Type": "default", "Parameter": "X_30", "Weight": None},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "graph_score", "Weight": None},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1, 2, 2],
        "Node": [0, 1, 0, 1],
        "X_1": [0.1, 0.2, 0.3, 0.4],
    })
    graph = pd.DataFrame({
        "Graph_ID": [1, 2],
        "X_30": [0.5, 0.7],
        "graph_score": [3.14, 2.71],  # continuous → regression
    })
    return _build_workbook({
        "Parameter": parameter,
        "Node": nodes,
        "Graph": graph,
    })


# ── parse_parameter_sheet ──────────────────────────────────────────


def test_parse_parameter_sheet_success():
    df = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label", "Weight": 2.5},
    ])
    spec = parse_parameter_sheet(df)
    assert len(spec.entries) == 2
    assert spec.y_levels() == ["Node"]
    assert spec.x_columns("Node", "default") == ["X_1"]
    assert spec.y_columns("Node", "default") == ["label"]
    payload = spec.to_payload()
    assert payload["entries"][1]["weight"] == 2.5


def test_parse_parameter_sheet_missing_columns():
    df = pd.DataFrame([{"XY": "X", "Level": "Node"}])
    with pytest.raises(ValueError, match="missing required columns"):
        parse_parameter_sheet(df)


def test_parse_parameter_sheet_invalid_xy():
    df = pd.DataFrame([
        {"XY": "Z", "Level": "Node", "Type": "default", "Parameter": "X_1"},
    ])
    with pytest.raises(ValueError, match="XY must be 'X' or 'Y'"):
        parse_parameter_sheet(df)


def test_parse_parameter_sheet_invalid_level():
    df = pd.DataFrame([
        {"XY": "X", "Level": "Hyperedge", "Type": "default", "Parameter": "X_1"},
    ])
    with pytest.raises(ValueError, match="Level must be"):
        parse_parameter_sheet(df)


def test_parse_parameter_sheet_blank_rows_skipped():
    df = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": None, "Level": None, "Type": None, "Parameter": None},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label"},
    ])
    spec = parse_parameter_sheet(df)
    assert len(spec.entries) == 2


def test_parse_parameter_sheet_y_blank_weight_defaults_to_one():
    """Blank Weight cell for a Y row → ParameterEntry.weight == 1.0 (not None)."""
    df = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label", "Weight": None},
    ])
    spec = parse_parameter_sheet(df)
    y_entry = next(e for e in spec.entries if e.xy == "Y")
    assert y_entry.weight == 1.0, f"Y weight default must be 1.0, got {y_entry.weight!r}"


def test_parse_parameter_sheet_y_no_weight_column_defaults_to_one():
    """Parameter sheet without a Weight column → Y rows still get weight 1.0."""
    df = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label"},
    ])
    spec = parse_parameter_sheet(df)
    y_entry = next(e for e in spec.entries if e.xy == "Y")
    assert y_entry.weight == 1.0


def test_parse_parameter_sheet_weight_non_numeric():
    df = pd.DataFrame([
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label", "Weight": "high"},
    ])
    with pytest.raises(ValueError, match="Weight must be numeric"):
        parse_parameter_sheet(df)


# ── parse_excel_file ──────────────────────────────────────────────


def test_parse_excel_node_classification():
    result = parse_excel_file(_node_y_classification_workbook(), "my-dataset")
    assert result["task_type"] == "node_classification"
    assert result["label_column"] == "label"
    assert result["label_weight"] == 2.0
    assert result["name"] == "my-dataset"
    assert "node_id" in result["nodes_df"].columns  # Node → node_id normalisation
    assert "src_id" in result["edges_df"].columns
    assert "dst_id" in result["edges_df"].columns
    assert len(result["nodes_df"]) == 5
    assert result["is_heterogeneous"] is False


def test_parse_excel_graph_regression():
    result = parse_excel_file(_graph_y_regression_workbook(), "g")
    assert result["task_type"] == "graph_regression"
    assert result["label_column"] == "graph_score"
    assert result["label_weight"] == 1.0  # default when Weight blank
    assert result["graph_df"] is not None
    assert result["is_heterogeneous"] is False


def test_parse_excel_missing_parameter_sheet():
    bad = _build_workbook({"Node": pd.DataFrame({"Node": [0]})})
    with pytest.raises(ValueError, match="missing the required 'Parameter' sheet"):
        parse_excel_file(bad)


def test_parse_excel_edge_y_deferred():
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Edge", "Type": "default", "Parameter": "edge_label"},
    ])
    nodes = pd.DataFrame({"Node": [0, 1], "X_1": [0.1, 0.2]})
    edges = pd.DataFrame({
        "Source_Node_ID": [0], "Target_Node_ID": [1], "edge_label": [1],
    })
    wb = _build_workbook({
        "Parameter": parameter, "Node": nodes, "Edge": edges,
    })
    with pytest.raises(ValueError, match="Edge-level prediction"):
        parse_excel_file(wb)


def test_parse_excel_multi_y_levels_deferred():
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "node_label"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "graph_label"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1],
        "Node": [0, 1],
        "X_1": [0.1, 0.2],
        "node_label": [0, 1],
    })
    graph = pd.DataFrame({"Graph_ID": [1], "graph_label": [1]})
    wb = _build_workbook({
        "Parameter": parameter,
        "Node": nodes,
        "Graph": graph,
    })
    with pytest.raises(ValueError, match="Multi-Y across different Levels"):
        parse_excel_file(wb)


# ── Multi-Y on a single Level (new in 2026-05-12) ──────────────────────

def test_parse_excel_multi_y_graph_regression():
    """Two continuous Y columns on Graph level → multi-Y regression."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "y1", "Weight": 2.0},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "y2", "Weight": 0.5},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1, 2, 2],
        "Node": [0, 1, 0, 1],
        "Type": ["default"] * 4,
        "X_1": [0.1, 0.2, 0.3, 0.4],
    })
    graph = pd.DataFrame({
        "Graph_ID": [1, 2],
        "Type": ["default", "default"],
        "y1": [3.14, 2.71],
        "y2": [1.41, 1.62],
    })
    wb = _build_workbook({
        "Parameter": parameter, "Node": nodes, "Graph": graph,
    })
    result = parse_excel_file(wb)
    assert result["task_type"] == "graph_regression"
    assert result["label_columns"] == ["y1", "y2"]
    assert result["label_weights"] == [2.0, 0.5]
    # Backwards-compat: singular fields still emitted.
    assert result["label_column"] == "y1"
    assert result["label_weight"] == 2.0


def test_parse_excel_multi_y_node_regression():
    """Two continuous Y columns on Node level → multi-Y node regression."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "delay", "Weight": 1.0},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "slack", "Weight": 3.0},
    ])
    nodes = pd.DataFrame({
        "Node": [0, 1, 2, 3],
        "Type": ["default"] * 4,
        "X_1": [0.1, 0.2, 0.3, 0.4],
        "delay": [0.15, 0.42, -0.33, 1.7],
        "slack": [-0.05, 0.12, 0.88, -0.4],
    })
    wb = _build_workbook({"Parameter": parameter, "Node": nodes})
    result = parse_excel_file(wb)
    assert result["task_type"] == "node_regression"
    assert result["label_columns"] == ["delay", "slack"]
    assert result["label_weights"] == [1.0, 3.0]


def test_parse_excel_multi_y_mixed_kinds_rejected():
    """Mixing regression + classification Y on the same Level is rejected."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "score"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "cls"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 2], "Node": [0, 0],
        "Type": ["default", "default"], "X_1": [0.1, 0.2],
    })
    graph = pd.DataFrame({
        "Graph_ID": [1, 2], "Type": ["default", "default"],
        "score": [3.14, 2.71],   # continuous
        "cls": [0, 1],            # integer few-uniques → classification
    })
    wb = _build_workbook({
        "Parameter": parameter, "Node": nodes, "Graph": graph,
    })
    with pytest.raises(ValueError, match="same kind"):
        parse_excel_file(wb)


def test_parse_excel_multi_y_classification_rejected():
    """Multi-Y classification is deferred to v2; raises with a clear message."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "cls_a"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "cls_b"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 2], "Node": [0, 0],
        "Type": ["default", "default"], "X_1": [0.1, 0.2],
    })
    graph = pd.DataFrame({
        "Graph_ID": [1, 2], "Type": ["default", "default"],
        "cls_a": [0, 1], "cls_b": [1, 0],
    })
    wb = _build_workbook({
        "Parameter": parameter, "Node": nodes, "Graph": graph,
    })
    with pytest.raises(ValueError, match="Multi-Y classification"):
        parse_excel_file(wb)


def test_parse_excel_no_y_raises():
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
    ])
    nodes = pd.DataFrame({"Node": [0], "X_1": [0.1]})
    wb = _build_workbook({"Parameter": parameter, "Node": nodes})
    with pytest.raises(ValueError, match="at least one Y row"):
        parse_excel_file(wb)


def test_parse_excel_missing_node_sheet():
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label"},
    ])
    wb = _build_workbook({"Parameter": parameter})
    with pytest.raises(ValueError, match="'Node' sheet"):
        parse_excel_file(wb)


def test_parse_excel_label_column_missing_in_data():
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label"},
    ])
    # Data sheet exists but lacks declared label column
    nodes = pd.DataFrame({"Node": [0], "X_1": [0.1]})
    wb = _build_workbook({"Parameter": parameter, "Node": nodes})
    with pytest.raises(ValueError, match="Label column 'label'"):
        parse_excel_file(wb)


def test_parse_excel_continuous_node_y_is_regression():
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "score"},
    ])
    nodes = pd.DataFrame({
        "Node": [0, 1, 2, 3],
        "X_1": [0.1, 0.2, 0.3, 0.4],
        "score": [0.15, 0.42, -0.33, 1.7],  # non-integer → regression
    })
    wb = _build_workbook({"Parameter": parameter, "Node": nodes})
    result = parse_excel_file(wb)
    assert result["task_type"] == "node_regression"


# ── New tests for simplified V2 schema (2026-04-25) ───────────────


def test_single_sheet_homogeneous():
    """Node/Edge/Graph sheets without Type column, single-type Parameter → success."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "feat_a"},
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "feat_b"},
        {"XY": "X", "Level": "Edge", "Type": "default", "Parameter": "weight"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "score"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1, 1],
        "Node": [0, 1, 2],
        "feat_a": [0.1, 0.2, 0.3],
        "feat_b": [1.0, 2.0, 3.0],
    })
    edges = pd.DataFrame({
        "Graph_ID": [1, 1],
        "Source_Node_ID": [0, 1],
        "Target_Node_ID": [1, 2],
        "weight": [0.5, 0.8],
    })
    graph = pd.DataFrame({"Graph_ID": [1], "score": [42.7]})
    wb = _build_workbook({
        "Parameter": parameter,
        "Node": nodes,
        "Edge": edges,
        "Graph": graph,
    })
    result = parse_excel_file(wb, "test-homo")
    assert result["task_type"] == "graph_regression"
    assert result["label_column"] == "score"
    assert result["is_heterogeneous"] is False
    assert len(result["nodes_df"]) == 3
    assert len(result["edges_df"]) == 2
    assert list(result["node_dfs"].keys()) == ["default"]
    assert list(result["edge_dfs"].keys()) == ["default"]
    assert result["canonical_edges"] == [("default", "default", "default")]
    assert result["node_dfs"]["default"]["_node_type"].unique().tolist() == ["default"]
    assert result["edge_dfs"]["default"]["_edge_type"].unique().tolist() == ["default"]


def test_in_sheet_type_column_splits_into_node_dfs():
    """Node sheet with 2 distinct Type values → node_dfs has 2 keys, is_heterogeneous=True."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "cell", "Parameter": "x"},
        {"XY": "X", "Level": "Node", "Type": "pin", "Parameter": "y"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "z"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1, 1],
        "Node": [0, 1, 2],
        "Type": ["cell", "cell", "pin"],
        "x": [0.1, 0.2, 0.0],
        "y": [0.0, 0.0, 0.5],
    })
    graph = pd.DataFrame({"Graph_ID": [1], "z": [0.5]})
    wb = _build_workbook({
        "Parameter": parameter, "Node": nodes, "Graph": graph,
    })
    result = parse_excel_file(wb)
    assert result["is_heterogeneous"] is True
    assert set(result["node_dfs"].keys()) == {"cell", "pin"}
    assert len(result["node_dfs"]["cell"]) == 2
    assert len(result["node_dfs"]["pin"]) == 1
    assert result["node_dfs"]["cell"]["_node_type"].unique().tolist() == ["cell"]
    assert result["node_dfs"]["pin"]["_node_type"].unique().tolist() == ["pin"]
    # Concatenated unified view has all rows
    assert len(result["nodes_df"]) == 3


def test_homogeneous_still_works_without_type_column():
    """Node/Edge sheets without Type column → single 'default' key, is_heterogeneous=False."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "feat_a"},
        {"XY": "X", "Level": "Edge", "Type": "default", "Parameter": "weight"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "score"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1, 1],
        "Node": [0, 1, 2],
        "feat_a": [0.1, 0.2, 0.3],
    })
    edges = pd.DataFrame({
        "Graph_ID": [1, 1],
        "Source_Node_ID": [0, 1],
        "Target_Node_ID": [1, 2],
        "weight": [0.5, 0.8],
    })
    graph = pd.DataFrame({"Graph_ID": [1], "score": [42.7]})
    wb = _build_workbook({
        "Parameter": parameter, "Node": nodes, "Edge": edges, "Graph": graph,
    })
    result = parse_excel_file(wb, "homo-no-type")
    assert result["is_heterogeneous"] is False
    assert list(result["node_dfs"].keys()) == ["default"]
    assert list(result["edge_dfs"].keys()) == ["default"]
    assert result["node_dfs"]["default"]["_node_type"].unique().tolist() == ["default"]
    assert result["edge_dfs"]["default"]["_edge_type"].unique().tolist() == ["default"]


def test_shared_feature_across_types():
    """Feature 'area' declared for both cell and pin types should populate
    in both per-type DataFrames and have no spurious missing counts.

    Audit confirmation: _split_by_type retains ALL columns from the unified
    sheet for each per-type sub-frame (only the Type column is dropped).
    A shared column like 'area' that exists in the unified sheet will therefore
    appear in BOTH the cell and pin DataFrames with their respective row values.
    """
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "cell", "Parameter": "area"},
        {"XY": "X", "Level": "Node", "Type": "pin",  "Parameter": "area"},
        {"XY": "X", "Level": "Node", "Type": "cell", "Parameter": "cell_only"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "score"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1, 1, 1],
        "Node": [0, 1, 2, 3],
        "Type": ["cell", "cell", "pin", "pin"],
        "area": [1.0, 2.0, 3.0, 4.0],
        "cell_only": [10.0, 20.0, None, None],
    })
    graph = pd.DataFrame({"Graph_ID": [1], "score": [42.5]})
    wb = _build_workbook({"Parameter": parameter, "Node": nodes, "Graph": graph})
    result = parse_excel_file(wb, "shared")
    assert result["is_heterogeneous"] is True
    assert set(result["node_dfs"].keys()) == {"cell", "pin"}
    # area is in BOTH per-type frames
    assert "area" in result["node_dfs"]["cell"].columns
    assert "area" in result["node_dfs"]["pin"].columns
    # cell_only column is present in the unified sheet, so it appears in both
    # sub-frames (the split only drops the Type column).  The pin rows have NaN
    # for cell_only, which is correct — type-scoped missing-count logic lives
    # above this layer and handles the NaN appropriately.
    assert "cell_only" in result["node_dfs"]["cell"].columns
    # Both per-type cell areas equal [1, 2]; pin areas equal [3, 4]
    assert list(result["node_dfs"]["cell"]["area"]) == [1.0, 2.0]
    assert list(result["node_dfs"]["pin"]["area"]) == [3.0, 4.0]
    # cell rows have real cell_only values; pin rows have NaN
    assert list(result["node_dfs"]["cell"]["cell_only"]) == [10.0, 20.0]
    assert result["node_dfs"]["pin"]["cell_only"].isna().all()


def test_canonical_edges_only_from_observed_triples():
    """canonical_edges should reflect only (src_type, rel, dst_type) actually
    present in edge data, not Cartesian product over node/edge types."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node",  "Type": "cell",    "Parameter": "f1"},
        {"XY": "X", "Level": "Node",  "Type": "pin",     "Parameter": "f2"},
        {"XY": "X", "Level": "Node",  "Type": "net",     "Parameter": "f3"},
        {"XY": "X", "Level": "Edge",  "Type": "cell_pin","Parameter": "ew"},
        {"XY": "X", "Level": "Edge",  "Type": "pin_net", "Parameter": "wl"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "y"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1] * 6,
        "Node":     [0, 1, 2, 3, 4, 5],
        "Type":     ["cell", "cell", "pin", "pin", "net", "net"],
        "f1": [1.0, 2.0, None, None, None, None],
        "f2": [None, None, 3.0, 4.0, None, None],
        "f3": [None, None, None, None, 5.0, 6.0],
    })
    edges = pd.DataFrame({
        "Graph_ID":      [1, 1, 1, 1],
        "Source_Node_ID":[0, 1, 2, 3],
        "Target_Node_ID":[2, 3, 4, 5],
        "Type":          ["cell_pin", "cell_pin", "pin_net", "pin_net"],
        "ew": [0.1, 0.2, None, None],
        "wl": [None, None, 10.0, 20.0],
    })
    graph = pd.DataFrame({"Graph_ID": [1], "y": [42.0]})
    wb = _build_workbook({
        "Parameter": parameter,
        "Node": nodes,
        "Edge": edges,
        "Graph": graph,
    })
    result = parse_excel_file(wb)
    assert result["is_heterogeneous"] is True
    triples = set(map(tuple, result["canonical_edges"]))
    assert triples == {("cell", "cell_pin", "pin"), ("pin", "pin_net", "net")}, \
        f"Expected only observed triples, got {triples}"


# ── Step 1 new tests (sheet optionality + type fallback + warnings) ───────────


def test_type_empty_fallback_default():
    """Parameter sheet row with empty Type → type_ becomes 'default', no raise."""
    df = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": None, "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": None, "Parameter": "label"},
    ])
    spec = parse_parameter_sheet(df)
    assert all(e.type_ == "default" for e in spec.entries)
    assert len(spec.entries) == 2


def test_validate_consistency_returns_warnings_not_raises():
    """Declared type not in data sheet → returns list with 1 warning, does not raise."""
    df = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "cell", "Parameter": "x"},
        {"XY": "X", "Level": "Node", "Type": "ghost", "Parameter": "y"},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "z"},
    ])
    spec = parse_parameter_sheet(df)
    # Only "cell" is observed in data sheet; "ghost" is declared but absent
    warnings = validate_hetero_consistency(spec, {"Node": ["cell"]})
    assert isinstance(warnings, list)
    assert len(warnings) == 1
    assert "ghost" in warnings[0]


def test_typo_fuzzy_match_within_distance():
    """Declared 'CAPP', observed 'CAP' → warning contains 'may be a typo for'."""
    df = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "CAPP", "Parameter": "f"},
        {"XY": "Y", "Level": "Node", "Type": "CAPP", "Parameter": "label"},
    ])
    spec = parse_parameter_sheet(df)
    warnings = validate_hetero_consistency(spec, {"Node": ["CAP"]})
    assert any("may be a typo for" in w for w in warnings), \
        f"Expected typo hint in warnings, got: {warnings}"


def test_no_edge_sheet_passes():
    """Workbook with only Node + Parameter sheet (no Edge) → parse succeeds."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label"},
    ])
    nodes = pd.DataFrame({
        "Node": [0, 1, 2],
        "X_1": [0.1, 0.2, 0.3],
        "label": [0, 1, 0],
    })
    wb = _build_workbook({"Parameter": parameter, "Node": nodes})
    result = parse_excel_file(wb)
    assert result["task_type"] == "node_classification"
    assert result["edge_dfs"] == {}


def test_no_graph_sheet_node_task():
    """No Graph sheet with node-level Y config → parse succeeds (graph_df is None)."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1"},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label"},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1, 1, 2, 2],
        "Node": [0, 1, 0, 1],
        "X_1": [0.1, 0.2, 0.3, 0.4],
        "label": [0, 1, 1, 0],
    })
    wb = _build_workbook({"Parameter": parameter, "Node": nodes})
    result = parse_excel_file(wb)
    assert result["task_type"] == "node_classification"
    assert result["graph_df"] is None
    assert result["schema_warnings"] == []
