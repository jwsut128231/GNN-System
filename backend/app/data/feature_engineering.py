"""Generic feature engineering + explore statistics for Excel-ingested data."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── column analysis ────────────────────────────────────────────────────────

def detect_column_type(series: pd.Series) -> str:
    """Auto-detect column type: 'numeric', 'categorical', or 'boolean'."""
    if series.dropna().empty:
        return "categorical"

    unique = set(series.dropna().unique())
    if unique <= {0, 1, True, False, "True", "False", "true", "false"}:
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 2:
            return "boolean"
        return "numeric"

    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().sum() > 0.8 * series.notna().sum():
        return "numeric"

    return "categorical"


# ── explore-page stats ─────────────────────────────────────────────────────

def _graph_stats(df: pd.DataFrame) -> tuple[int, float]:
    """Return (graph_count, avg_rows_per_graph) based on _graph column."""
    if "_graph" in df.columns and len(df) > 0:
        groups = df["_graph"].value_counts()
        return int(len(groups)), round(float(groups.mean()), 2) if len(groups) > 0 else 0.0
    return 1, float(len(df))


NODE_COL_SKIP = {"_graph", "_node_type", "Type", "Graph_ID", "Node", "node_id"}
EDGE_COL_SKIP = {"_graph", "_edge_type", "src_type", "dst_type",
                 "Edge_Type", "Type", "src_id", "dst_id",
                 "Graph_ID", "Source_Node_ID", "Target_Node_ID",
                 "Source_Node_Type", "Target_Node_Type"}


def _column_entries(
    df: pd.DataFrame,
    skip: set[str],
    *,
    type_name: Optional[str] = None,
    source: str = "node",
    declared_cols: Optional[set[str]] = None,
    graph_col: str = "_graph",
) -> list[dict]:
    """Emit a list of column-stat dicts from ``df``.

    When ``type_name`` is provided the ``node_type`` / ``edge_type`` field is
    attached so the UI can group columns by the type they belong to. Missing
    counts are computed against ``len(df)`` — the rows of this *type* only —
    so heterogeneous graphs don't over-report missing values when concatenated.

    When ``declared_cols`` is provided (heterogeneous mode), only columns that
    are in that set are included. This prevents cross-type padding columns
    (e.g. ``f_pin`` present in cell rows as all-NaN) from being reported as
    100 % missing for the wrong type.

    Per-graph aware missing semantics (2026-04-28):
        When ``graph_col`` is in ``df``, a column is considered "in scope" only
        for the graphs that actually use it (i.e. have at least one non-NaN
        value for that column). Rows in graphs that legitimately don't carry
        that column are excluded from both numerator and denominator. This
        mirrors the user-facing rule "if this type has A&B or A&C, an A&C
        row should NOT report B as missing" — only true value gaps inside an
        in-scope graph are counted.
    """
    out = []
    has_graph = graph_col in df.columns and len(df) > 0
    if has_graph:
        graph_ids = df[graph_col].dropna().unique().tolist()

    for col in df.columns:
        if col in skip:
            continue
        if declared_cols is not None and col not in declared_cols:
            continue
        series = df[col]

        if has_graph:
            # Restrict denominator to rows in graphs that USE this column.
            using_graphs = [
                gid for gid in graph_ids
                if df.loc[df[graph_col] == gid, col].notna().any()
            ]
            in_scope_mask = df[graph_col].isin(using_graphs)
            in_scope = df.loc[in_scope_mask, col]
            total_in_scope = len(in_scope)
            missing_in_scope = int(in_scope.isna().sum())
            n_total_graphs = len(graph_ids)
            n_using_graphs = len(using_graphs)
            graph_presence_pct = round(
                (n_using_graphs / n_total_graphs * 100) if n_total_graphs else 0.0, 2
            )
        else:
            in_scope = series
            total_in_scope = len(df)
            missing_in_scope = int(series.isna().sum())
            graph_presence_pct = 100.0

        if has_graph and total_in_scope == 0:
            # No graph uses this column at all — entirely missing, not an
            # in-scope NaN: surface as 0% so the UI can flag it clearly.
            missing_pct = 100.0
            presence_pct = 0.0
        else:
            missing_pct = round(
                (missing_in_scope / total_in_scope * 100) if total_in_scope else 0.0, 2
            )
            presence_pct = round(100.0 - missing_pct, 2)
        entry = {
            "name": col,
            "dtype": detect_column_type(series),
            "missing_count": missing_in_scope,
            "missing_pct": missing_pct,
            "presence_pct": presence_pct,
            "graph_presence_pct": graph_presence_pct,
            "low_presence_warning": False,
            "unique_count": int(series.nunique()),
        }
        if type_name is not None:
            entry[f"{source}_type"] = type_name
        out.append(entry)
    return out


def compute_per_graph_feature_schema(
    node_dfs: dict[str, pd.DataFrame],
    graph_col: str = "_graph",
    min_presence_ratio: float = 0.1,
) -> dict[str, dict]:
    """Compute per-type, per-graph feature schema with presence rates.

    Returns:
    {
      "CAP": {
        "graphs": { "g1": ["A", "B"], "g2": ["A", "C"] },
        "union": ["A", "B", "C"],
        "intersection": ["A"],
        "presence_per_column": { "A": 1.0, "B": 0.5, "C": 0.5 },
        "low_presence_columns": [],
      },
      ...
    }

    A column is considered "present" in a graph if at least one row for that
    (type, graph) combination has a non-NaN value.
    ``low_presence_columns`` lists columns whose graph-level presence ratio is
    below ``min_presence_ratio``.
    """
    skip = NODE_COL_SKIP
    result: dict[str, dict] = {}

    for node_type, df in node_dfs.items():
        # Determine feature columns (skip internal/meta columns)
        feature_cols = [c for c in df.columns if c not in skip]

        # Get unique graph IDs
        if graph_col in df.columns:
            graph_ids = df[graph_col].dropna().unique().tolist()
        else:
            graph_ids = ["__single__"]

        # Build per-graph column lists: columns with at least one non-NaN row
        graphs: dict[str, list[str]] = {}
        for gid in graph_ids:
            if graph_col in df.columns:
                g_df = df[df[graph_col] == gid]
            else:
                g_df = df
            present = [c for c in feature_cols if g_df[c].notna().any()]
            graphs[str(gid)] = present

        # Union and intersection
        all_sets = [set(cols) for cols in graphs.values()] if graphs else []
        union_set: list[str] = sorted(set().union(*all_sets)) if all_sets else []
        intersection_set: list[str] = sorted(set.intersection(*all_sets)) if all_sets else []

        # Presence ratio per column = fraction of graphs where column is present
        n_graphs = len(graphs)
        presence_per_column: dict[str, float] = {}
        for col in union_set:
            present_count = sum(1 for cols in graphs.values() if col in cols)
            presence_per_column[col] = round(present_count / n_graphs, 4) if n_graphs else 0.0

        low_presence_columns = [
            col for col, ratio in presence_per_column.items()
            if ratio < min_presence_ratio
        ]

        result[node_type] = {
            "graphs": graphs,
            "union": union_set,
            "intersection": intersection_set,
            "presence_per_column": presence_per_column,
            "low_presence_columns": low_presence_columns,
        }

    return result


def compute_generic_explore(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    is_heterogeneous: bool = False,
    node_types: Optional[list[str]] = None,
    edge_types: Optional[list[str]] = None,
    canonical_edges: Optional[list] = None,
    node_dfs: Optional[dict[str, pd.DataFrame]] = None,
    edge_dfs: Optional[dict[str, pd.DataFrame]] = None,
    node_type_features: Optional[dict[str, list[str]]] = None,
    edge_type_features: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Compute generic exploration stats for an Excel-ingested dataset.

    Includes multi-graph summary (graph_count + avg_nodes_per_graph +
    avg_edges_per_graph) and heterogeneity metadata when available.

    When ``node_dfs`` / ``edge_dfs`` are supplied (per-type DataFrames), the
    column-stats block is computed PER TYPE so missing counts use the row
    count of that type as the denominator. Without per-type input, stats fall
    back to the unified DataFrames (legacy / homogeneous behaviour).

    ``node_type_features`` / ``edge_type_features`` map each type name to the
    list of X column names declared for it in the Parameter sheet.  When
    supplied, only declared columns are included in per-type stats, preventing
    cross-type NaN padding columns from being flagged as 100 % missing.
    """
    # ── Column stats ──
    # In heterogeneous mode with per-type inputs, each (type, column) is a
    # separate entry. In homogeneous mode, we use the unified frame.
    if node_dfs:
        columns: list[dict] = []
        for t, df in node_dfs.items():
            declared = (
                set(node_type_features[t]) if node_type_features and t in node_type_features
                else None
            )
            columns.extend(_column_entries(
                df, NODE_COL_SKIP, type_name=t, source="node",
                declared_cols=declared,
            ))
    else:
        columns = _column_entries(nodes_df, NODE_COL_SKIP)

    numeric_cols = [c["name"] for c in columns
                    if c["dtype"] == "numeric" and c["name"].lower() not in ("node_id", "id", "index")]
    # Deduplicate — a feature may appear under multiple types with the same name.
    seen: set[str] = set()
    numeric_cols = [n for n in numeric_cols if not (n in seen or seen.add(n))]
    if len(numeric_cols) > 5:
        variances = nodes_df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = list(variances.index[:5])
    correlation = compute_correlation(nodes_df, numeric_cols) if numeric_cols else []

    if edge_dfs:
        edge_columns: list[dict] = []
        for t, df in edge_dfs.items():
            declared = (
                set(edge_type_features[t]) if edge_type_features and t in edge_type_features
                else None
            )
            edge_columns.extend(_column_entries(
                df, EDGE_COL_SKIP, type_name=t, source="edge",
                declared_cols=declared,
            ))
    else:
        edge_columns = _column_entries(edges_df, EDGE_COL_SKIP)

    graph_count, avg_nodes = _graph_stats(nodes_df)
    _, avg_edges = _graph_stats(edges_df)

    # Per-graph feature schema (hetero only)
    if is_heterogeneous and node_dfs:
        per_graph_schema = compute_per_graph_feature_schema(node_dfs)
    else:
        per_graph_schema = {}

    payload = {
        "num_nodes": len(nodes_df),
        "num_edges": len(edges_df),
        "columns": columns,
        "edge_columns": edge_columns,
        "feature_correlation": correlation,
        "correlation_columns": numeric_cols,
        # Multi-graph & heterogeneity summary
        "graph_count": graph_count,
        "avg_nodes_per_graph": avg_nodes,
        "avg_edges_per_graph": avg_edges,
        "is_heterogeneous": bool(is_heterogeneous),
        "node_types": node_types or [],
        "edge_types": edge_types or [],
        "canonical_edges": [list(ce) for ce in (canonical_edges or [])],
        "per_graph_feature_schema": per_graph_schema,
    }
    return payload


def analyze_numeric_column(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "column": series.name, "dtype": "numeric",
            "mean": 0, "median": 0, "std": 0,
            "min": 0, "max": 0, "q1": 0, "q3": 0,
            "outlier_count": 0, "distribution": [],
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    outliers = int(((clean < lo) | (clean > hi)).sum())

    counts, bin_edges = np.histogram(clean, bins=10)
    distribution = [
        {"range": f"{round(float(bin_edges[i]), 4)}~{round(float(bin_edges[i + 1]), 4)}",
         "count": int(counts[i])}
        for i in range(len(counts))
    ]

    return {
        "column": series.name, "dtype": "numeric",
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "q1": round(q1, 4), "q3": round(q3, 4),
        "outlier_count": outliers, "distribution": distribution,
    }


def analyze_categorical_column(series: pd.Series) -> dict:
    vc = series.dropna().value_counts()
    value_counts = [{"name": str(k), "count": int(v)} for k, v in vc.items()]
    top_value = str(vc.index[0]) if len(vc) > 0 else ""
    top_count = int(vc.iloc[0]) if len(vc) > 0 else 0
    return {
        "column": series.name, "dtype": "categorical",
        "value_counts": value_counts,
        "top_value": top_value, "top_count": top_count,
    }


def compute_correlation(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    if not columns:
        return []
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return []
    numeric = df[valid].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    out = []
    for xi in valid:
        for yi in valid:
            v = corr.loc[xi, yi]
            out.append({"x": xi, "y": yi, "value": round(float(v), 2) if not pd.isna(v) else 0.0})
    return out


# ── Label validation + imputation ──────────────────────────────────────────

def validate_label(nodes_df: pd.DataFrame, column: str, task_type: str) -> dict:
    if column not in nodes_df.columns:
        return {"valid": False, "message": f"Column '{column}' not found in dataset."}

    series = nodes_df[column]
    missing = int(series.isna().sum())
    if missing > 0:
        return {
            "valid": False,
            "message": f"Label column '{column}' has {missing} missing values. Please impute first.",
        }

    if task_type in ("node_classification", "graph_classification"):
        unique_vals = series.unique()
        n = len(unique_vals)
        if n < 2:
            return {"valid": False, "message": f"Classification requires at least 2 classes, found {n}."}
        if n > 100:
            return {"valid": False, "message": f"Too many classes ({n}). Consider regression instead."}
        vc = series.value_counts()
        return {
            "valid": True,
            "message": f"Valid classification target with {n} classes.",
            "num_classes": n,
            "class_distribution": [{"label": str(k), "count": int(v)} for k, v in vc.items()],
        }

    if task_type in ("node_regression", "graph_regression"):
        numeric_series = pd.to_numeric(series, errors="coerce")
        non_numeric = int(numeric_series.isna().sum() - series.isna().sum())
        if non_numeric > 0:
            return {"valid": False, "message": f"Regression target has {non_numeric} non-numeric values."}
        return {
            "valid": True,
            "message": f"Valid regression target.",
            "is_continuous": series.nunique() > 10,
            "value_range": {
                "min": round(float(numeric_series.min()), 4),
                "max": round(float(numeric_series.max()), 4),
                "mean": round(float(numeric_series.mean()), 4),
                "std": round(float(numeric_series.std()), 4),
            },
        }

    return {"valid": False, "message": f"Unknown task type: {task_type}"}


def impute_column(df: pd.DataFrame, column: str, method: str) -> tuple[pd.DataFrame, int]:
    if column not in df.columns:
        return df, 0
    mask = df[column].isna()
    filled = int(mask.sum())
    if filled == 0:
        return df, 0
    df = df.copy()
    if method == "mean":
        fill = pd.to_numeric(df[column], errors="coerce").mean()
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fill)
    elif method == "median":
        fill = pd.to_numeric(df[column], errors="coerce").median()
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fill)
    elif method == "zero":
        df[column] = df[column].fillna(0)
    else:
        return df, 0
    return df, filled
