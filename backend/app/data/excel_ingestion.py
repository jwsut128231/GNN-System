"""Excel (.xlsx) graph-data ingestion.

Reads the multi-sheet template (Parameter + Node/Edge/Graph data sheets),
derives the user's intended task from Y rows in the Parameter sheet, and emits
DataFrames ready for the PyG converter.

Schema (V3 — 2026-04-26):

    One sheet per level:  ``Node``, ``Edge``, ``Graph``.
    Data sheets MAY contain a ``Type`` column:
        - Absent OR all values equal  → homogeneous (single key "default").
        - Present with multiple distinct values → heterogeneous; rows are split
          into per-type DataFrames keyed by type name.
    The Parameter sheet carries a ``Type`` column and may declare multiple Type
    values per Level for heterogeneous graphs.

Scope:
    * Homogeneous and heterogeneous graphs.
    * Y must be declared on exactly one Level (Node or Graph).
    * Edge-level prediction (Y on Edge) is still deferred.
"""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd

from app.data.excel_spec import (
    ExcelGraphSpec,
    VALID_LEVELS,
    parse_parameter_sheet,
    validate_hetero_consistency,
)


# ── Column-name normalisation ──
NODE_ID_CANDIDATES = ("Node", "node_id", "NodeID", "Node_ID", "node")
SRC_ID_CANDIDATES = ("Source_Node_ID", "src_id", "source", "Source", "SourceNodeID")
DST_ID_CANDIDATES = ("Target_Node_ID", "dst_id", "target", "Target", "TargetNodeID")
GRAPH_ID_CANDIDATES = ("Graph_ID", "graph_id", "GraphID")
TYPE_COL_CANDIDATES = ("Type", "type", "TYPE", "node_type", "edge_type")


def _pick(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _require(df: pd.DataFrame, candidates: tuple[str, ...], sheet: str, label: str) -> str:
    col = _pick(df, candidates)
    if col is None:
        raise ValueError(
            f"Sheet '{sheet}' missing required {label} column "
            f"(looked for any of {list(candidates)})."
        )
    return col


def _infer_task_kind(series: pd.Series) -> str:
    """Return 'classification' or 'regression' based on a Y column's values."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "classification"
    nunique = clean.nunique()
    is_integer = bool(((clean.astype(float) % 1) == 0).all())
    if is_integer and nunique <= 20:
        return "classification"
    return "regression"


def _validate_scope(spec: ExcelGraphSpec) -> None:
    """Enforce the scope boundary for the current implementation."""
    y_levels = spec.y_levels()
    if not y_levels:
        raise ValueError(
            "Parameter sheet must declare at least one Y row "
            "(to indicate the prediction target)."
        )
    if "Edge" in y_levels:
        raise ValueError(
            "Edge-level prediction (Y on Edge) is not yet supported."
        )
    if len(y_levels) > 1:
        raise ValueError(
            f"Multi-Y across different Levels is not yet supported; Y declared "
            f"on multiple levels: {y_levels}. Place all Y columns on the same "
            f"Level."
        )


def _load_workbook(source: bytes | str) -> dict[str, pd.DataFrame]:
    buf: io.BytesIO | str
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        sheets = pd.read_excel(buf, sheet_name=None, engine="openpyxl")
    except ImportError as e:
        raise ValueError(
            "openpyxl is required to read .xlsx files. Install with 'pip install openpyxl'."
        ) from e
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}") from e
    if "Parameter" not in sheets:
        raise ValueError(
            "Excel workbook is missing the required 'Parameter' sheet."
        )
    return sheets


def _normalise_node_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    out = df.copy()
    node_col = _require(out, NODE_ID_CANDIDATES, sheet_name, "node id")
    if node_col != "node_id":
        out = out.rename(columns={node_col: "node_id"})
    out["node_id"] = out["node_id"].astype(str)
    g_col = _pick(out, GRAPH_ID_CANDIDATES)
    if g_col and g_col != "_graph":
        out = out.rename(columns={g_col: "_graph"})
    return out


def _normalise_edge_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    out = df.copy()
    src_col = _require(out, SRC_ID_CANDIDATES, sheet_name, "source node id")
    dst_col = _require(out, DST_ID_CANDIDATES, sheet_name, "target node id")
    renames = {}
    if src_col != "src_id":
        renames[src_col] = "src_id"
    if dst_col != "dst_id":
        renames[dst_col] = "dst_id"
    if renames:
        out = out.rename(columns=renames)
    out["src_id"] = out["src_id"].astype(str)
    out["dst_id"] = out["dst_id"].astype(str)

    g_col = _pick(out, GRAPH_ID_CANDIDATES)
    if g_col and g_col != "_graph":
        out = out.rename(columns={g_col: "_graph"})
    return out


def _split_by_type(
    df: pd.DataFrame,
    type_col: Optional[str],
    internal_col: str,
    default_label: str = "default",
) -> tuple[dict[str, pd.DataFrame], bool]:
    """Split *df* by the values in *type_col*.

    Returns:
        (per_type_dict, is_hetero)
        - per_type_dict: keyed by type string; each sub-frame has *internal_col*
          set to the type string and the original type column dropped.
        - is_hetero: True when more than one distinct type is present.
    """
    if type_col is None:
        # No Type column → homogeneous
        out = df.copy()
        out[internal_col] = default_label
        return {default_label: out}, False

    values = df[type_col].fillna(default_label).astype(str)
    distinct = values.unique().tolist()

    if len(distinct) <= 1:
        # All rows share the same type → treat as homogeneous
        label = distinct[0] if distinct else default_label
        out = df.drop(columns=[type_col]).copy()
        out[internal_col] = label
        return {label: out}, False

    # Multiple types → heterogeneous split
    per_type: dict[str, pd.DataFrame] = {}
    for t in distinct:
        sub = df[values == t].drop(columns=[type_col]).copy()
        sub[internal_col] = t
        per_type[t] = sub.reset_index(drop=True)

    return per_type, True


def parse_excel_file(source: bytes | str, dataset_name: str = "") -> dict:
    """Parse an Excel workbook matching graph_data_template.xlsx.

    Expected sheets: ``Parameter``, ``Node``, ``Edge`` (optional), ``Graph`` (optional).
    Data sheets may contain a ``Type`` column:
        - Absent or single-valued → homogeneous (``is_heterogeneous=False``).
        - Multi-valued → heterogeneous (``is_heterogeneous=True``).

    Returns:
        dict with keys:
            spec                 : ExcelGraphSpec
            is_heterogeneous     : bool
            nodes_df             : node DataFrame (concatenated, with ``_node_type``)
            edges_df             : edge DataFrame (concatenated, with ``_edge_type``)
            graph_df             : Optional[pd.DataFrame]
            node_dfs             : dict[node_type, DataFrame]
            edge_dfs             : dict[edge_type, DataFrame]
            canonical_edges      : list[tuple[src_type, rel, dst_type]]
            task_type            : e.g. "graph_regression"
            label_column         : Y column name
            label_weight         : float (default 1.0)
            name                 : dataset_name
    """
    sheets = _load_workbook(source)
    spec = parse_parameter_sheet(sheets["Parameter"])
    _validate_scope(spec)

    # ── Load Node sheet (required) ──
    if "Node" not in sheets:
        raise ValueError(
            "Excel workbook is missing the required 'Node' sheet. "
            "Please provide a sheet named exactly 'Node'."
        )
    node_norm = _normalise_node_sheet(sheets["Node"], "Node")
    node_type_col = _pick(node_norm, TYPE_COL_CANDIDATES)
    # Exclude the canonical id/graph columns from being mistaken for Type.
    # _pick searches by name so node_id / _graph won't match TYPE_COL_CANDIDATES.
    node_dfs, node_is_hetero = _split_by_type(node_norm, node_type_col, "_node_type")

    # ── Load Edge sheet (optional) ──
    edge_dfs: dict[str, pd.DataFrame] = {}
    edge_is_hetero = False
    if "Edge" in sheets:
        edge_norm = _normalise_edge_sheet(sheets["Edge"], "Edge")
        edge_type_col = _pick(edge_norm, TYPE_COL_CANDIDATES)
        edge_dfs, edge_is_hetero = _split_by_type(edge_norm, edge_type_col, "_edge_type")

    is_heterogeneous = node_is_hetero or edge_is_hetero

    # ── Validate in-sheet types against Parameter sheet declarations ──
    node_in_sheet_types = list(node_dfs.keys()) if node_is_hetero else []
    edge_in_sheet_types = list(edge_dfs.keys()) if edge_is_hetero else []
    schema_warnings = validate_hetero_consistency(spec, {
        "Node": node_in_sheet_types,
        "Edge": edge_in_sheet_types,
    })

    # ── Load Graph sheet (optional) ──
    graph_df: Optional[pd.DataFrame] = None
    if "Graph" in sheets:
        graph_df = sheets["Graph"].copy()
        gcol = _pick(graph_df, GRAPH_ID_CANDIDATES)
        if gcol and gcol != "_graph":
            graph_df = graph_df.rename(columns={gcol: "_graph"})

    if not node_dfs:
        raise ValueError(
            "Parameter sheet must declare at least one Node-level entry "
            "(to identify graph vertices)."
        )

    # Unified views (concatenate per-type frames).
    unified_nodes = pd.concat(list(node_dfs.values()), ignore_index=True)
    unified_edges = (
        pd.concat(list(edge_dfs.values()), ignore_index=True)
        if edge_dfs
        else pd.DataFrame(columns=["src_id", "dst_id"])
    )

    # Canonical edges: derive from observed (src_type, edge_type, dst_type) triples
    # rather than a Cartesian product over all node/edge type combinations.
    canonical_edges: list[tuple[str, str, str]] = []
    if edge_dfs:
        if is_heterogeneous:
            # Build node_id -> _node_type lookup from the unified nodes frame.
            node_type_lookup: dict[str, str] = dict(
                zip(
                    unified_nodes["node_id"].astype(str),
                    unified_nodes["_node_type"].astype(str),
                )
            )
            seen: set[tuple[str, str, str]] = set()
            for et, edf in edge_dfs.items():
                for _, row in edf.iterrows():
                    s = node_type_lookup.get(str(row["src_id"]))
                    d = node_type_lookup.get(str(row["dst_id"]))
                    if s is None or d is None:
                        continue
                    triple = (s, et, d)
                    if triple not in seen:
                        seen.add(triple)
                        canonical_edges.append(triple)
        else:
            node_t = next(iter(node_dfs))
            edge_t = next(iter(edge_dfs))
            canonical_edges.append((node_t, edge_t, node_t))

    # ── derive task_type + label_columns (multi-Y aware) ──
    y_level = spec.y_levels()[0]   # "Node" or "Graph"

    # Collect every Y entry on this level (across all Types) so multi-Y on a
    # single Level produces parallel regression targets.
    y_entries_all = [e for e in spec.entries if e.xy == "Y" and e.level == y_level]
    label_columns: list[str] = [e.parameter for e in y_entries_all]
    label_weights: list[float] = [
        float(e.weight) if e.weight is not None else 1.0
        for e in y_entries_all
    ]

    def _source_df_for_y(col: str) -> pd.DataFrame:
        if y_level == "Node":
            for _t, _df in node_dfs.items():
                if col in _df.columns:
                    return _df
            raise ValueError(
                f"Label column '{col}' declared in Parameter sheet "
                f"is not present in the Node sheet."
            )
        # Graph
        if graph_df is None or col not in graph_df.columns:
            raise ValueError(
                f"Label column '{col}' declared in Parameter sheet "
                f"is not present in the Graph sheet."
            )
        return graph_df

    kinds: list[str] = []
    for col in label_columns:
        src_df = _source_df_for_y(col)
        kinds.append(_infer_task_kind(src_df[col]))

    if len(set(kinds)) > 1:
        mixed = list(zip(label_columns, kinds))
        raise ValueError(
            f"All Y columns must be the same kind (regression or classification); "
            f"got mixed kinds: {mixed}."
        )
    kind = kinds[0]
    if len(label_columns) > 1 and kind == "classification":
        raise ValueError(
            "Multi-Y classification is not yet supported; only multi-Y "
            "regression is supported in this release."
        )
    task_type = f"{y_level.lower()}_{kind}"

    # Backwards-compatible singular fields (first Y).
    label_column = label_columns[0]
    label_weight = label_weights[0]

    return {
        "spec": spec,
        "is_heterogeneous": is_heterogeneous,
        "nodes_df": unified_nodes,
        "edges_df": unified_edges,
        "graph_df": graph_df,
        "node_dfs": node_dfs,
        "edge_dfs": edge_dfs,
        "canonical_edges": canonical_edges,
        "task_type": task_type,
        "label_column": label_column,
        "label_weight": label_weight,
        "label_columns": label_columns,
        "label_weights": label_weights,
        "name": dataset_name or "excel-upload",
        "schema_warnings": schema_warnings,
    }
