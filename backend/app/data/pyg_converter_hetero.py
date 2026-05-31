"""Heterogeneous graph conversion — DataFrames → torch_geometric.data.HeteroData.

Consumes the output of ``excel_ingestion.parse_excel_file`` for heterogeneous
graphs. Each node Type becomes a ``HeteroData`` node type; each edge Type (with
its (src_type, rel, dst_type) canonical triple) becomes a relation.

This module focuses on **graph regression / graph classification** where ``y``
is a single scalar per graph taken from the Graph_{type} sheet, aligned by
``_graph``. Node-level heterogeneous prediction is deferred.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# Shared ToUndirected transform — appends a reverse edge for every relation
# so node types without incoming edges still receive messages under to_hetero().
_TO_UNDIRECTED = T.ToUndirected(merge=False)


def _numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _fit_scalers(
    node_dfs: dict[str, pd.DataFrame],
    exclude: set[str],
    min_presence_ratio: float = 0.1,
) -> tuple[dict[str, StandardScaler], dict[str, list[str]], dict[str, list[str]]]:
    """Fit StandardScaler per node type.

    Columns whose non-NaN ratio across all rows of that type is below
    ``min_presence_ratio`` are excluded from scaling (but kept in feature_cols
    so the tensor shape stays consistent — they are fillna(0.0) without scaling).

    Returns:
        scalers            : per-type fitted StandardScaler (fitted on scale_cols only)
        feature_cols       : per-type full list of numeric feature columns (union)
        excluded_cols      : per-type list of columns excluded from scaling
    """
    scalers: dict[str, StandardScaler] = {}
    feature_cols: dict[str, list[str]] = {}
    excluded_cols: dict[str, list[str]] = {}

    for t, df in node_dfs.items():
        cols = _numeric_feature_columns(df, exclude)
        feature_cols[t] = cols

        excl: list[str] = []
        scale_cols: list[str] = []
        for c in cols:
            n_total = len(df)
            n_present = df[c].notna().sum()
            ratio = n_present / n_total if n_total > 0 else 0.0
            if ratio < min_presence_ratio:
                excl.append(c)
            else:
                scale_cols.append(c)

        excluded_cols[t] = excl

        if scale_cols:
            vals = df[scale_cols].fillna(0.0).to_numpy(dtype=np.float32)
            sc = StandardScaler()
            sc.fit(vals)
            scalers[t] = sc
        else:
            scalers[t] = StandardScaler()

    return scalers, feature_cols, excluded_cols


def _build_single_hetero(
    graph_id,
    node_dfs: dict[str, pd.DataFrame],
    edge_dfs: dict[str, pd.DataFrame],
    graph_df: Optional[pd.DataFrame],
    label_columns: list[str],
    canonical_edges: list[tuple[str, str, str]],
    scalers: dict[str, StandardScaler],
    feature_cols: dict[str, list[str]],
    excluded_cols: Optional[dict[str, list[str]]] = None,
) -> HeteroData:
    """Build one HeteroData for a single graph_id.

    ``label_columns`` carries one or more Y columns; the resulting ``data.y``
    has shape ``[1]`` for single-Y and ``[T]`` for multi-Y.
    """
    data = HeteroData()
    T = len(label_columns)

    # node types + id → local index maps per type
    id_maps: dict[str, dict] = {}
    for nt, df in node_dfs.items():
        sub = df[df["_graph"] == graph_id] if "_graph" in df.columns else df
        sub = sub.reset_index(drop=True)
        id_map = {str(v): i for i, v in enumerate(sub["node_id"].tolist())}
        id_maps[nt] = id_map

        cols = feature_cols[nt]
        excl = set((excluded_cols or {}).get(nt, []))
        if cols:
            raw = sub[cols].fillna(0.0).to_numpy(dtype=np.float32)
            # Only scale columns that passed the min_presence_ratio threshold
            scale_cols = [c for c in cols if c not in excl]
            if scale_cols and len(raw):
                scale_idx = [cols.index(c) for c in scale_cols]
                scaled_part = scalers[nt].transform(raw[:, scale_idx]).astype(np.float32)
                vals = raw.copy()
                for i, si in enumerate(scale_idx):
                    vals[:, si] = scaled_part[:, i]
            else:
                vals = raw
        else:
            vals = np.zeros((len(sub), 1), dtype=np.float32)
        data[nt].x = torch.tensor(vals, dtype=torch.float)
        data[nt].num_nodes = len(sub)

    # edge types
    for src_t, rel, dst_t in canonical_edges:
        edf = edge_dfs[rel]
        sub = edf[edf["_graph"] == graph_id] if "_graph" in edf.columns else edf
        sub = sub.reset_index(drop=True)
        src_idx = []
        dst_idx = []
        for _, row in sub.iterrows():
            s = str(row["src_id"])
            d = str(row["dst_id"])
            if s in id_maps[src_t] and d in id_maps[dst_t]:
                src_idx.append(id_maps[src_t][s])
                dst_idx.append(id_maps[dst_t][d])
        if src_idx:
            edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        data[src_t, rel, dst_t].edge_index = edge_index

    # graph-level label(s) — skip when graph_df not available (e.g. node-level tasks)
    if graph_df is not None:
        row = graph_df[graph_df["_graph"] == graph_id]
        if row.empty:
            raise ValueError(
                f"Graph sheet has no row for graph {graph_id} (labels {label_columns})."
            )
        vals = [float(row[c].iloc[0]) for c in label_columns]
        if T == 1:
            data.y = torch.tensor([vals[0]], dtype=torch.float)
        else:
            # Shape (1, T) so PyG batching concatenates to (B, T) at the graph level.
            data.y = torch.tensor([vals], dtype=torch.float)
    data.num_targets = T

    # Append reverse edges so every node type appears as a destination.
    data = _TO_UNDIRECTED(data)
    return data


def parsed_excel_to_hetero_list(
    parsed: dict,
) -> tuple[list[HeteroData], dict[str, StandardScaler], dict[str, list[str]], list[tuple[str, str, str]], dict[str, list[str]]]:
    """Convert a parse_excel_file() result into a list of HeteroData (one per graph).

    Supports multi-Y: when ``parsed`` carries ``label_columns`` (list[str]) the
    resulting HeteroData.y for each graph is a vector of length T. The legacy
    ``label_column`` key is honoured as a fallback.

    Returns:
        data_list, scalers, feature_names_by_type, metadata_edges, excluded_cols
    metadata_edges is a list of canonical (src_type, relation, dst_type) tuples —
    together with sorted(scalers.keys()) this forms the HeteroData metadata needed
    by ``to_hetero()``. excluded_cols maps node type → list of columns skipped by
    the scaler due to low presence ratio (stored alongside scalers for inference).
    """
    node_dfs = parsed["node_dfs"]
    edge_dfs = parsed["edge_dfs"]
    graph_df = parsed["graph_df"]
    label_columns = list(parsed.get("label_columns") or [parsed["label_column"]])
    task_type = parsed.get("task_type", "")
    canonical_edges = parsed["canonical_edges"]

    if graph_df is None and task_type.startswith("graph"):
        raise ValueError(
            "Heterogeneous graph_regression / graph_classification requires a "
            "Graph-level sheet with the Y column; none was provided."
        )

    exclude = {"node_id", "_graph", "_node_type", "_edge_type",
               "src_id", "dst_id", "src_type", "dst_type", "Graph_ID",
               "Type", "Edge_Type"}

    scalers, feature_cols, excluded_cols = _fit_scalers(node_dfs, exclude)

    graph_ids = sorted({
        gid for df in node_dfs.values() if "_graph" in df.columns
        for gid in df["_graph"].dropna().unique().tolist()
    })
    if not graph_ids:
        # Single implicit graph — fall back to graph_df or a synthetic id
        if graph_df is not None:
            graph_ids = [graph_df["_graph"].iloc[0] if "_graph" in graph_df.columns else 1]
        else:
            graph_ids = [1]

    # For node-level tasks (no graph-level Y), explicitly skip the Y
    # lookup. For graph-level tasks (or when task_type is omitted by older
    # callers), pass graph_df through so per-graph labels are set.
    pass_graph_df = graph_df if not task_type.startswith("node") else None

    data_list: list[HeteroData] = []
    for gid in graph_ids:
        d = _build_single_hetero(
            gid, node_dfs, edge_dfs,
            pass_graph_df,
            label_columns,
            canonical_edges, scalers, feature_cols,
            excluded_cols=excluded_cols,
        )
        data_list.append(d)

    return data_list, scalers, feature_cols, canonical_edges, excluded_cols
