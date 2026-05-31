"""Homogeneous DataFrame → PyG Data conversion.

Two public entry points:
    * ``dataframes_to_pyg_dynamic`` — single Data for node-level tasks.
    * ``dataframes_to_graph_list``  — list[Data], one per Graph_ID, for
      graph-level tasks on multi-graph datasets.

Both entry points accept ``label_columns`` (list[str]) and build a
multi-target ``y`` tensor when ``len(label_columns) > 1``. When ``T == 1`` the
shape is unchanged from the single-Y era so older callers keep working.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from app.data.feature_engineering import detect_column_type


_NODE_EXCLUDE = {"node_id", "_graph", "_node_type", "Graph_ID", "Type"}
_EDGE_EXCLUDE = {"src_id", "dst_id", "_graph", "_edge_type",
                 "src_type", "dst_type", "Graph_ID", "Edge_Type"}


def _as_label_list(label_columns: Union[str, Sequence[str]]) -> list[str]:
    """Normalise the label_columns arg: accept str (single Y) or list."""
    if isinstance(label_columns, str):
        return [label_columns]
    return list(label_columns)


def _numeric_feature_matrix(
    df: pd.DataFrame,
    exclude: set[str],
    scaler: StandardScaler | None,
    fit_scaler: bool,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    cols = [c for c in df.columns if c not in exclude and detect_column_type(df[c]) == "numeric"]
    if not cols:
        return np.zeros((len(df), 1), dtype=np.float32), scaler or StandardScaler(), []
    vals = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        vals = scaler.fit_transform(vals).astype(np.float32) if len(vals) else vals
    elif scaler is not None and len(vals):
        vals = scaler.transform(vals).astype(np.float32)
    return vals, scaler, cols


def dataframes_to_pyg_dynamic(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    label_column: Union[str, Sequence[str]],
    task_type: str,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[Data, StandardScaler, list[str]]:
    """Convert unified node/edge DataFrames to a single Data object.

    Used for node-level tasks. For multi-graph graph-level tasks, use
    ``dataframes_to_graph_list`` instead.

    ``label_column`` accepts a single column name (legacy) or a list of column
    names for multi-target training. When more than one Y column is supplied
    the resulting ``y`` tensor has shape ``(N, T)`` for regression.
    """
    label_columns = _as_label_list(label_column)
    T = len(label_columns)
    exclude = set(_NODE_EXCLUDE) | set(label_columns)
    x_vals, scaler, feature_names = _numeric_feature_matrix(nodes_df, exclude, scaler, fit_scaler)

    id_to_idx = {str(v): i for i, v in enumerate(nodes_df["node_id"].tolist())}

    # Labels
    is_regression = task_type.endswith("regression")
    if is_regression:
        # Build (N, T) matrix. When T == 1 we squeeze to (N,) to preserve the
        # historical single-Y shape so older code paths keep working untouched.
        cols_arr = np.stack(
            [
                pd.to_numeric(nodes_df[c], errors="coerce").fillna(0.0)
                .to_numpy(dtype=np.float32)
                for c in label_columns
            ],
            axis=1,
        )
        if T == 1:
            y = torch.tensor(cols_arr.squeeze(-1), dtype=torch.float)
        else:
            y = torch.tensor(cols_arr, dtype=torch.float)
        num_classes = 1
    else:
        # Classification is single-Y only (multi-Y classification is deferred,
        # so upstream validation rejects T > 1 here).
        y_raw = nodes_df[label_columns[0]]
        uniq = sorted(y_raw.dropna().unique().tolist())
        label_map = {v: i for i, v in enumerate(uniq)}
        y_int = y_raw.map(label_map).fillna(0).to_numpy(dtype=np.int64)
        y = torch.tensor(y_int, dtype=torch.long)
        num_classes = len(uniq) or 2

    # Edges
    src_idx, dst_idx = [], []
    for s, d in zip(edges_df.get("src_id", []), edges_df.get("dst_id", [])):
        if s is None or d is None:
            continue
        si, di = str(s), str(d)
        if si in id_to_idx and di in id_to_idx:
            src_idx.append(id_to_idx[si])
            dst_idx.append(id_to_idx[di])
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long) if src_idx \
        else torch.zeros((2, 0), dtype=torch.long)

    # Edge features
    edge_feat, _, _ = _numeric_feature_matrix(edges_df, _EDGE_EXCLUDE, None, True)
    edge_attr = torch.tensor(edge_feat[: edge_index.shape[1]], dtype=torch.float) \
        if edge_index.shape[1] > 0 else torch.zeros((0, 1), dtype=torch.float)

    data = Data(
        x=torch.tensor(x_vals, dtype=torch.float),
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(nodes_df),
    )
    data.num_classes = num_classes
    data.num_features_actual = data.x.shape[1]
    data.num_targets = T
    return data, scaler, feature_names


def dataframes_to_graph_list(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    graph_df: Optional[pd.DataFrame],
    label_column: Union[str, Sequence[str]],
    task_type: str,
    scalers: Optional[dict] = None,
    fit_scaler: bool = True,
) -> tuple[list[Data], dict, list[str], int]:
    """Split a multi-graph DataFrame pair into a list of Data (one per graph).

    Each graph's ``y`` is a scalar (single-Y, ``shape=[1]``) or vector
    (multi-Y, ``shape=[T]``) taken from ``graph_df`` aligned by ``_graph``.

    Returns:
        (data_list, scalers_dict, feature_names, num_classes)
    """
    label_columns = _as_label_list(label_column)
    T = len(label_columns)

    if "_graph" not in nodes_df.columns:
        # Treat as single graph
        nodes_df = nodes_df.copy()
        nodes_df["_graph"] = 1
        if not edges_df.empty:
            edges_df = edges_df.copy()
            edges_df["_graph"] = 1

    graph_ids = sorted(nodes_df["_graph"].dropna().unique().tolist())

    # Fit single scaler on all numeric node features
    exclude = set(_NODE_EXCLUDE) | set(label_columns)
    x_all, node_scaler, feature_names = _numeric_feature_matrix(
        nodes_df, exclude, (scalers or {}).get("node"), fit_scaler,
    )
    scalers = {"node": node_scaler}

    is_regression = task_type.endswith("regression")
    if is_regression:
        num_classes = 1
        label_map = None
    else:
        primary = label_columns[0]
        if graph_df is not None and primary in graph_df.columns:
            uniq = sorted(graph_df[primary].dropna().unique().tolist())
        else:
            uniq = sorted(nodes_df[primary].dropna().unique().tolist())
        num_classes = len(uniq) or 2
        label_map = {v: i for i, v in enumerate(uniq)}

    def _y_for_graph(gid, sub_nodes: pd.DataFrame) -> torch.Tensor:
        """Return the y tensor for a single graph_id.

        Regression: shape [1] when T == 1, else [T].
        Classification: shape [1] (single-Y only).
        """
        if is_regression:
            vals: list[float] = []
            for col in label_columns:
                v = None
                if graph_df is not None and col in graph_df.columns \
                        and "_graph" in graph_df.columns:
                    row = graph_df[graph_df["_graph"] == gid]
                    if not row.empty:
                        v = row[col].iloc[0]
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    series = sub_nodes.get(col)
                    if series is None:
                        raise ValueError(
                            f"Label '{col}' not found in graph {gid} "
                            f"(neither Graph sheet nor Node sheet)."
                        )
                    v = float(pd.to_numeric(series, errors="coerce").mean())
                vals.append(float(v))
            if T == 1:
                return torch.tensor([vals[0]], dtype=torch.float)
            # Shape (1, T) so PyG batching concatenates to (B, T).
            return torch.tensor([vals], dtype=torch.float)

        # Classification — single Y guaranteed (T == 1).
        col = label_columns[0]
        v = None
        if graph_df is not None and col in graph_df.columns \
                and "_graph" in graph_df.columns:
            row = graph_df[graph_df["_graph"] == gid]
            if not row.empty:
                v = row[col].iloc[0]
        if v is None:
            series = sub_nodes.get(col)
            if series is None:
                raise ValueError(
                    f"Label '{col}' not found in graph {gid} "
                    f"(neither Graph sheet nor Node sheet)."
                )
            v = series.mode().iloc[0]
        return torch.tensor([label_map.get(v, 0)], dtype=torch.long)

    data_list: list[Data] = []
    for gid in graph_ids:
        node_mask = nodes_df["_graph"] == gid
        sub_nodes = nodes_df[node_mask].reset_index(drop=True)
        sub_x = x_all[node_mask.to_numpy()]
        id_map = {str(v): i for i, v in enumerate(sub_nodes["node_id"].tolist())}

        # Edges in this graph
        if not edges_df.empty and "_graph" in edges_df.columns:
            sub_edges = edges_df[edges_df["_graph"] == gid]
        else:
            sub_edges = edges_df
        src, dst = [], []
        for s, d in zip(sub_edges.get("src_id", []), sub_edges.get("dst_id", [])):
            if s is None or d is None:
                continue
            si, di = str(s), str(d)
            if si in id_map and di in id_map:
                src.append(id_map[si])
                dst.append(id_map[di])
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src \
            else torch.zeros((2, 0), dtype=torch.long)

        y_t = _y_for_graph(gid, sub_nodes)
        data = Data(
            x=torch.tensor(sub_x, dtype=torch.float),
            y=y_t,
            edge_index=edge_index,
            num_nodes=len(sub_nodes),
        )
        data.num_targets = T
        data_list.append(data)

    return data_list, scalers, feature_names, num_classes
