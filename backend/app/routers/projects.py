"""Project API — Excel-driven workflow.

Endpoints cover the full Upload → Explore → Train → Report pipeline plus the
model registry. All data ingress goes through the Excel template path; legacy
CSV endpoints have been removed.
"""
from __future__ import annotations

import io
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import (
    APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, UploadFile,
)
from fastapi.responses import Response, StreamingResponse

from app.core import store
from app.core.config import settings
from app.data import graph_cache_sqlite
from app.data.excel_ingestion import parse_excel_file
from app.data.feature_engineering import (
    analyze_categorical_column,
    analyze_numeric_column,
    compute_correlation,
    compute_generic_explore,
    detect_column_type,
    impute_column,
    validate_label,
)
from app.schemas.api_models import (
    ConfirmDataRequest,
    CorrelationRequest,
    CreateExperimentRequest,
    CreateProjectRequest,
    UpdateProjectRequest,
    DatasetSummary,
    ExperimentDetail,
    ExperimentSummary,
    GenericExploreData,
    ImputationRequest,
    ImputationResult,
    LabelValidationRequest,
    LabelValidationResult,
    ProjectDetail,
    ProjectSummary,
    RegisteredModel,
    RegisterModelRequest,
    StartTrainingRequest,
    TaskStatus,
    TrainingEstimate,
    Report,
)
from app.training.pipeline import run_training_task

router = APIRouter(prefix="/projects", tags=["projects"])


# ── helpers ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_or_404(project_id: str) -> dict:
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _dataset_for_project(project: dict) -> dict:
    ds_id = project.get("dataset_id")
    if not ds_id:
        raise HTTPException(status_code=400, detail="No dataset uploaded for this project")
    ds = store.get_dataset(ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


def _task_to_status(task: dict, project_id: str) -> TaskStatus:
    return TaskStatus(
        task_id=task["task_id"], project_id=project_id, status=task["status"],
        progress=task.get("progress", 0),
        current_phase=task.get("current_phase"),
        current_trial=task.get("current_trial"),
        total_trials=task.get("total_trials"),
        device=task.get("device"),
        results=task.get("results"),
        best_config=task.get("best_config"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
    )


def _to_summary(p: dict) -> ProjectSummary:
    return ProjectSummary(
        project_id=p["project_id"], name=p["name"], tags=p.get("tags", []),
        created_at=p["created_at"], updated_at=p.get("updated_at", p["created_at"]),
        current_step=p.get("current_step", 1), status=p.get("status", "created"),
        dataset_id=p.get("dataset_id"), task_id=p.get("task_id"),
    )


def _dataset_to_summary(ds: dict) -> DatasetSummary:
    return DatasetSummary(
        dataset_id=ds["dataset_id"], name=ds["name"],
        num_nodes=ds["num_nodes"], num_edges=ds["num_edges"],
        num_features=ds.get("num_features", 0),
        num_classes=ds.get("num_classes", 0),
        is_directed=ds.get("is_directed", True),
        task_type=ds.get("task_type", "graph_regression"),
        declared_task_type=ds.get("declared_task_type"),
        declared_label_column=ds.get("declared_label_column"),
        schema_spec=ds.get("schema_spec"),
        graph_count=ds.get("graph_count", 1),
        is_heterogeneous=ds.get("is_heterogeneous", False),
        node_types=ds.get("node_types", []),
        edge_types=ds.get("edge_types", []),
        label_columns=ds.get("label_columns") or (
            [ds["label_column"]] if ds.get("label_column") else []
        ),
        label_weights=ds.get("label_weights") or (
            [ds["label_weight"]] if ds.get("label_weight") is not None else []
        ),
    )


# ── CRUD ───────────────────────────────────────────────────────────────────

@router.post("/", response_model=ProjectSummary)
async def create_project(body: CreateProjectRequest):
    project_id = str(uuid.uuid4())
    now = _now_iso()
    record = {
        "project_id": project_id, "name": body.name, "tags": body.tags,
        "created_at": now, "updated_at": now,
        "current_step": 1, "status": "created",
        "dataset_id": None, "task_type": None, "label_column": None,
        "imputation_log": [], "training_config": None,
        "task_id": None, "task_ids": [],
    }
    store.put_project(project_id, record)
    return _to_summary(record)


@router.get("/", response_model=list[ProjectSummary])
async def list_projects_endpoint():
    return [_to_summary(p) for p in store.list_projects()]


# ── Demo Excel + Template download (must precede /{project_id} routes) ────

DEMO_EXCELS = [
    {
        "id": "multigraph_homo",
        "name": "Multi-Graph Homogeneous",
        "description": "30 graphs, single node/edge type, graph_regression target (target_delay)",
        "filename": "demo_multigraph_homo.v2.xlsx",
        "is_heterogeneous": False,
        "tags": ["multi-graph", "homogeneous", "graph-regression"],
    },
    {
        "id": "multigraph_homo_no_type",
        "name": "Multi-Graph Homogeneous (no Type column)",
        "description": "30 graphs, homogeneous, Node/Edge/Graph sheets WITHOUT a Type column — auto-detected as homogeneous",
        "filename": "demo_multigraph_homo_no_type.xlsx",
        "is_heterogeneous": False,
        "tags": ["multi-graph", "homogeneous", "graph-regression", "no-type-column"],
    },
    {
        "id": "multigraph_multi_y",
        "name": "Multi-Graph Multi-Y Regression",
        "description": "30 graphs, homogeneous, two Y targets (target_delay weight=2.0 + target_power_mw weight=1.0 default), no Type column",
        "filename": "demo_multigraph_multi_y.xlsx",
        "is_heterogeneous": False,
        "tags": ["multi-graph", "homogeneous", "graph-regression", "multi-y", "no-type-column"],
    },
    {
        "id": "multigraph_hetero",
        "name": "Multi-Graph Heterogeneous",
        "description": "30 graphs, 3 node types (cell/pin/net), 3 edge types, graph_regression target (total_wirelength)",
        "filename": "demo_multigraph_hetero.v2.xlsx",
        "is_heterogeneous": True,
        "tags": ["multi-graph", "heterogeneous", "graph-regression"],
    },
    {
        "id": "hetero_multifeature_str",
        "name": "Hetero Multi-Feature (string Graph_ID)",
        "description": (
            "30 graphs, string Graph_IDs (G001..G030), 2 node types (CAP/RES). "
            "CAP nodes carry per-graph variable feature subsets "
            "(X_1+X_2, X_1+X_3, X_2+X_3); target_y is a deterministic "
            "linear function so loss is stable and metrics are sane."
        ),
        "filename": "demo_hetero_multifeature.v3.xlsx",
        "is_heterogeneous": True,
        "tags": ["multi-graph", "heterogeneous", "graph-regression", "string-graph-id", "multi-feature"],
    },
]


@router.get("/demo-excels")
async def list_demo_excels():
    """List bundled demo Excel files available for one-click loading."""
    return DEMO_EXCELS


@router.get("/sample-excel")
async def download_sample_excel():
    """Return the empty graph_data_template.xlsx for users to fill in."""
    template_path = Path(__file__).resolve().parent.parent.parent / "graph_data_template.xlsx"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Excel template not found on server")
    return StreamingResponse(
        io.BytesIO(template_path.read_bytes()),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=graph_data_template.xlsx"},
    )


@router.get("/demo-excel/{demo_id}")
async def download_demo_excel(demo_id: str):
    """Download a bundled demo .xlsx by id."""
    match = next((d for d in DEMO_EXCELS if d["id"] == demo_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Demo '{demo_id}' not found")
    path = settings.DEMO_DATA_DIR / match["filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Demo Excel missing on disk; run scripts/generate_excel_demos.py")
    return StreamingResponse(
        io.BytesIO(path.read_bytes()),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename={match["filename"]}'},
    )


@router.post("/{project_id}/load-demo-excel", response_model=DatasetSummary)
async def load_demo_excel(project_id: str, demo_id: str = Query(...)):
    """One-click load a bundled demo .xlsx into a project."""
    _project_or_404(project_id)
    match = next((d for d in DEMO_EXCELS if d["id"] == demo_id), None)
    if not match:
        raise HTTPException(status_code=400, detail=f"Unknown demo_id '{demo_id}'")
    path = settings.DEMO_DATA_DIR / match["filename"]
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Demo Excel missing on disk; run scripts/generate_excel_demos.py",
        )
    return await _store_excel_dataset(project_id, path.read_bytes(), match["name"])


# ── Upload Excel ──────────────────────────────────────────────────────────

def _build_graph_index(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> list[dict]:
    """Return [{id, node_count, edge_count}, ...] per graph (quick pass)."""
    if "_graph" not in nodes_df.columns:
        return [{"id": "default", "node_count": len(nodes_df), "edge_count": len(edges_df)}]

    node_counts = nodes_df.groupby("_graph").size().rename("node_count")
    if "_graph" in edges_df.columns and not edges_df.empty:
        edge_counts = edges_df.groupby("_graph").size().rename("edge_count")
    else:
        edge_counts = pd.Series(dtype=int, name="edge_count")

    all_graphs = node_counts.index.union(edge_counts.index)
    result = []
    for g in all_graphs:
        result.append({
            "id": str(g),
            "node_count": int(node_counts.get(g, 0)),
            "edge_count": int(edge_counts.get(g, 0)),
        })
    return result


async def _store_excel_dataset(project_id: str, content: bytes, name: str) -> DatasetSummary:
    # Compute content hash for ETag / cache invalidation
    excel_hash = graph_cache_sqlite.content_hash(content)

    # Invalidate SQLite cache if re-upload with different content
    existing_project = store.get_project(project_id)
    if existing_project:
        old_ds_id = existing_project.get("dataset_id")
        if old_ds_id:
            old_ds = store.get_dataset(old_ds_id)
            if old_ds and old_ds.get("excel_hash") != excel_hash:
                graph_cache_sqlite.invalidate(old_ds_id)

    try:
        parsed = parse_excel_file(content, name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    nodes_df = parsed["nodes_df"]
    edges_df = parsed["edges_df"]
    graph_df = parsed["graph_df"]

    if nodes_df.empty:
        raise HTTPException(status_code=422, detail="Node data sheet contains no rows.")
    if "node_id" not in nodes_df.columns or nodes_df["node_id"].isna().any():
        raise HTTPException(status_code=422, detail="Every Node row must have a non-empty node id.")

    is_hetero = parsed["is_heterogeneous"]
    node_types = parsed["spec"].node_types()
    edge_types = parsed["spec"].edge_types()
    _spec = parsed["spec"]
    _node_type_features = {t: _spec.x_columns("Node", t) for t in node_types} if is_hetero else None
    _edge_type_features = {t: _spec.x_columns("Edge", t) for t in edge_types} if is_hetero else None

    explore_stats = compute_generic_explore(
        nodes_df, edges_df,
        is_heterogeneous=is_hetero,
        node_types=node_types, edge_types=edge_types,
        canonical_edges=parsed["canonical_edges"],
        node_dfs=parsed["node_dfs"] if is_hetero else None,
        edge_dfs=parsed["edge_dfs"] if is_hetero else None,
        node_type_features=_node_type_features,
        edge_type_features=_edge_type_features,
    )
    explore_stats["schema_warnings"] = parsed.get("schema_warnings", []) or []
    # Count unique numeric feature names (per-type entries with same name → one feature).
    _numeric_names = {c["name"] for c in explore_stats["columns"] if c["dtype"] == "numeric"}
    num_features = len(_numeric_names)

    dataset_id = str(uuid.uuid4())
    task_type = parsed["task_type"]
    label_column = parsed["label_column"]
    schema_payload = parsed["spec"].to_payload()

    # For node-level tasks we still pre-split train/test at ingest time so the
    # existing pipeline code path works. For graph-level tasks the pipeline
    # itself builds the per-graph list and splits.
    if task_type.startswith("node"):
        num_nodes = len(nodes_df)
        perm = torch.randperm(num_nodes)
        split = max(int(num_nodes * 0.8), 1)
        tr_idx = perm[:split].numpy()
        te_idx = perm[split:].numpy() if split < num_nodes else perm[:1].numpy()
        nodes_df_train = nodes_df.iloc[tr_idx].reset_index(drop=True)
        nodes_df_test = nodes_df.iloc[te_idx].reset_index(drop=True)
        train_ids = set(nodes_df_train["node_id"].values)
        test_ids = set(nodes_df_test["node_id"].values)
        if not edges_df.empty:
            edges_df_train = edges_df[
                edges_df["src_id"].isin(train_ids) & edges_df["dst_id"].isin(train_ids)
            ].reset_index(drop=True)
            edges_df_test = edges_df[
                edges_df["src_id"].isin(test_ids) & edges_df["dst_id"].isin(test_ids)
            ].reset_index(drop=True)
        else:
            edges_df_train = edges_df.copy()
            edges_df_test = edges_df.copy()
    else:
        nodes_df_train = nodes_df_test = nodes_df
        edges_df_train = edges_df_test = edges_df

    graph_index = _build_graph_index(nodes_df, edges_df)

    ds_record = {
        "dataset_id": dataset_id, "name": name,
        "num_nodes": len(nodes_df), "num_edges": len(edges_df),
        "num_features": num_features, "num_classes": 0,
        "is_directed": True, "task_type": task_type,
        # Unified DataFrames (for explore + homogeneous training)
        "nodes_df": nodes_df, "edges_df": edges_df, "graph_df": graph_df,
        # Pre-split (used only by node-level pipeline branch)
        "nodes_df_train": nodes_df_train, "nodes_df_test": nodes_df_test,
        "edges_df_train": edges_df_train, "edges_df_test": edges_df_test,
        # Heterogeneous decomposition
        "is_heterogeneous": is_hetero,
        "node_dfs": parsed["node_dfs"], "edge_dfs": parsed["edge_dfs"],
        "canonical_edges": parsed["canonical_edges"],
        "node_types": node_types, "edge_types": edge_types,
        "explore_stats": explore_stats,
        "graph_count": explore_stats["graph_count"],
        # Excel schema persistence
        "declared_task_type": task_type,
        "declared_label_column": label_column,
        "schema_spec": schema_payload,
        "label_column": label_column,
        "label_weight": parsed["label_weight"],
        # Multi-Y support: parallel lists; for single-Y these are length-1.
        "label_columns": parsed.get("label_columns") or [label_column],
        "label_weights": parsed.get("label_weights") or [parsed["label_weight"]],
        # Cache / ETag support
        "excel_hash": excel_hash,
        "graph_index": graph_index,
    }
    store.put_dataset(dataset_id, ds_record)

    project = store.get_project(project_id)
    ds_ids = project.get("dataset_ids", [])
    if dataset_id not in ds_ids:
        ds_ids.append(dataset_id)
    store.update_project(
        project_id, dataset_id=dataset_id, dataset_ids=ds_ids,
        task_type=task_type, label_column=label_column,
        current_step=3, status="data_confirmed", updated_at=_now_iso(),
    )

    return _dataset_to_summary(ds_record)


@router.post("/{project_id}/upload-excel", response_model=DatasetSummary)
async def upload_project_excel(
    project_id: str,
    file: UploadFile = File(...),
    dataset_name: str = Form(default=""),
):
    """Upload a graph_data_template.xlsx — single upload path for the platform."""
    _project_or_404(project_id)
    content = await file.read()
    name = dataset_name or (file.filename or "excel-upload").rsplit(".", 1)[0]
    return await _store_excel_dataset(project_id, content, name)


# ── Project detail ─────────────────────────────────────────────────────────

@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str):
    p = _project_or_404(project_id)
    detail = ProjectDetail(
        project_id=p["project_id"], name=p["name"], tags=p.get("tags", []),
        created_at=p["created_at"],
        current_step=p.get("current_step", 1),
        status=p.get("status", "created"),
        dataset_id=p.get("dataset_id"), task_id=p.get("task_id"),
        task_type=p.get("task_type"), label_column=p.get("label_column"),
        dataset_ids=p.get("dataset_ids", [p["dataset_id"]] if p.get("dataset_id") else []),
        experiment_ids=p.get("experiment_ids", []),
    )
    if p.get("dataset_id"):
        ds = store.get_dataset(p["dataset_id"])
        if ds:
            detail.dataset_summary = _dataset_to_summary(ds)
    if p.get("task_id"):
        task = store.get_task(p["task_id"])
        if task:
            detail.task_status = _task_to_status(task, project_id)
    return detail


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    _project_or_404(project_id)
    store.delete_project(project_id)
    return {"detail": "Project deleted"}


@router.patch("/{project_id}", response_model=ProjectSummary)
async def update_project(project_id: str, body: UpdateProjectRequest):
    _project_or_404(project_id)
    updates: dict = {"updated_at": _now_iso()}
    if body.name is not None:
        updates["name"] = body.name
    if body.tags is not None:
        updates["tags"] = body.tags
    store.update_project(project_id, **updates)
    return _to_summary(store.get_project(project_id))


# ── Explore ────────────────────────────────────────────────────────────────

@router.get("/{project_id}/explore", response_model=GenericExploreData)
async def explore_project_data(project_id: str):
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)
    stats = ds.get("explore_stats")
    if not stats:
        raise HTTPException(status_code=400, detail="Explore stats not computed yet")
    return stats


@router.get("/{project_id}/graph-sample")
async def get_graph_sample(
    request: Request,
    project_id: str,
    limit: int = Query(default=500, ge=5, le=5000),
    graph_name: Optional[str] = Query(default=None),
):
    """Return a sample of the project's graph data for preview.

    Returns node_type / edge_type for every item so the frontend can colour
    heterogeneous graphs. Supports ETag / 304 and SQLite caching per graph.
    """
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)
    dataset_id = ds["dataset_id"]
    excel_hash = ds.get("excel_hash", "")

    # Default to the first graph on multi-graph datasets so the initial
    # request doesn't fall back to a cross-graph BFS sample (which would
    # briefly render every graph stacked on top of each other).
    if not graph_name:
        gi = ds.get("graph_index", [])
        if len(gi) > 1:
            graph_name = str(gi[0]["id"])

    # ETag: encode dataset content + graph selection + limit
    etag = f'"{excel_hash}-{graph_name or "all"}-{limit}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)

    # SQLite cache hit (only when a specific graph is selected).
    # Cache key includes limit to avoid returning a 500-row sample for a
    # later 100-row request (and vice-versa) -- see security-review #1.
    cache_graph_id = f"{graph_name}|limit={limit}" if graph_name else None
    if cache_graph_id and excel_hash:
        cached = graph_cache_sqlite.get(dataset_id, cache_graph_id, excel_hash)
        if cached is not None:
            return Response(
                content=cached,
                media_type="application/json",
                headers={"ETag": etag},
            )

    nodes_df = ds["nodes_df"]
    edges_df = ds["edges_df"]
    graph_names = sorted(nodes_df["_graph"].dropna().unique().tolist()) \
        if "_graph" in nodes_df.columns else []
    graph_names = [str(g) for g in graph_names]

    if graph_name and "_graph" in nodes_df.columns:
        # Match as string for stable URL params
        nodes_df = nodes_df[nodes_df["_graph"].astype(str) == str(graph_name)].reset_index(drop=True)
        if "_graph" in edges_df.columns:
            edges_df = edges_df[edges_df["_graph"].astype(str) == str(graph_name)].reset_index(drop=True)

    sample_size = min(limit, len(nodes_df))
    all_node_ids = set(nodes_df["node_id"].values) if not nodes_df.empty else set()

    # Build adjacency using vectorized pandas instead of iterrows
    adj: dict = defaultdict(set)
    if not edges_df.empty and "src_id" in edges_df.columns:
        valid_mask = (
            edges_df["src_id"].isin(all_node_ids) & edges_df["dst_id"].isin(all_node_ids)
        )
        valid_edges = edges_df[valid_mask][["src_id", "dst_id"]]
        for rec in valid_edges.to_dict(orient="records"):
            s, d = rec["src_id"], rec["dst_id"]
            adj[s].add(d)
            adj[d].add(s)

    if not nodes_df.empty:
        seed = nodes_df.sample(n=1, random_state=42)["node_id"].values[0]
        sampled_ids = {seed}
        frontier = {seed}
        while len(sampled_ids) < sample_size and frontier:
            new_nodes = set()
            for nid in frontier:
                for nb in adj.get(nid, set()):
                    if nb not in sampled_ids:
                        new_nodes.add(nb)
            if not new_nodes:
                remaining = all_node_ids - sampled_ids
                if remaining:
                    import random as _r
                    rng = _r.Random(42)
                    pick = rng.sample(list(remaining), min(5, len(remaining)))
                    new_nodes = set(pick)
                else:
                    break
            frontier = set()
            for nid in new_nodes:
                if len(sampled_ids) >= sample_size:
                    break
                sampled_ids.add(nid)
                frontier.add(nid)
    else:
        sampled_ids = set()

    sampled = nodes_df[nodes_df["node_id"].isin(sampled_ids)] if sampled_ids else nodes_df.head(0)
    sampled_edges = edges_df[
        edges_df["src_id"].isin(sampled_ids) & edges_df["dst_id"].isin(sampled_ids)
    ] if not edges_df.empty and "src_id" in edges_df.columns else edges_df.head(0)

    def _norm_id(v):
        try:
            f = float(v)
            if f == int(f):
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return str(v)

    attr_cols = [c for c in nodes_df.columns if c not in {"node_id", "id", "index"}]
    nodes_out = []
    for _, row in sampled.iterrows():
        attrs = {}
        for c in attr_cols:
            v = row[c]
            if pd.isna(v):
                attrs[c] = None
            elif isinstance(v, (int, float, np.integer, np.floating)):
                attrs[c] = round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
            else:
                attrs[c] = str(v)
        nodes_out.append({
            "id": _norm_id(row["node_id"]),
            "label": _norm_id(row["node_id"]),
            "node_type": str(row["_node_type"]) if "_node_type" in row and not pd.isna(row["_node_type"]) else None,
            "attributes": attrs,
        })

    # Vectorized edge attribute extraction
    edge_attr_cols = [c for c in edges_df.columns if c not in {"src_id", "dst_id", "id", "index"}]
    edges_out = []
    for rec in sampled_edges.to_dict(orient="records"):
        attrs = {}
        for c in edge_attr_cols:
            v = rec.get(c)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                attrs[c] = None
            elif isinstance(v, (int, float, np.integer, np.floating)):
                attrs[c] = round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
            else:
                attrs[c] = str(v)
        edge_type_val = rec.get("_edge_type")
        edges_out.append({
            "source": _norm_id(rec["src_id"]),
            "target": _norm_id(rec["dst_id"]),
            "edge_type": str(edge_type_val) if edge_type_val is not None and not (isinstance(edge_type_val, float) and pd.isna(edge_type_val)) else None,
            "attributes": attrs,
        })

    payload = {
        "nodes": nodes_out, "edges": edges_out,
        "num_nodes_total": len(nodes_df),
        "num_edges_total": len(edges_df),
        "sample_size": sample_size,
        "graph_names": graph_names,
        "current_graph": graph_name,
        "is_heterogeneous": ds.get("is_heterogeneous", False),
        "node_types": ds.get("node_types", []),
        "edge_types": ds.get("edge_types", []),
        "graph_index": ds.get("graph_index", []),
    }

    payload_bytes = json.dumps(payload).encode()

    # Store in SQLite cache when a specific graph was requested
    if cache_graph_id and excel_hash:
        graph_cache_sqlite.put(dataset_id, cache_graph_id, excel_hash, payload_bytes)

    return Response(
        content=payload_bytes,
        media_type="application/json",
        headers={"ETag": etag},
    )


@router.get("/{project_id}/columns/{column_name}")
async def analyze_column(
    project_id: str, column_name: str,
    override_type: Optional[str] = Query(None),
):
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)
    nodes_df = ds["nodes_df"]
    if column_name not in nodes_df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found")
    series = nodes_df[column_name]
    col_type = override_type or detect_column_type(series)
    if col_type == "numeric":
        return analyze_numeric_column(series)
    return analyze_categorical_column(series)


@router.post("/{project_id}/correlation")
async def get_correlation_endpoint(project_id: str, body: CorrelationRequest):
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)
    return compute_correlation(ds["nodes_df"], body.columns)


@router.post("/{project_id}/validate-label", response_model=LabelValidationResult)
async def validate_label_endpoint(project_id: str, body: LabelValidationRequest):
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)
    # For graph-level tasks the label sits on graph_df; for node-level on nodes_df
    df = ds.get("graph_df") if body.task_type.startswith("graph") else ds["nodes_df"]
    if df is None:
        df = ds["nodes_df"]
    return validate_label(df, body.label_column, body.task_type)


@router.post("/{project_id}/impute", response_model=ImputationResult)
async def impute_missing_endpoint(project_id: str, body: ImputationRequest):
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)
    ds["nodes_df"], filled = impute_column(ds["nodes_df"], body.column, body.method)
    # Also update matching per-type frame(s) so hetero explore stats stay in sync.
    _is_hetero = ds.get("is_heterogeneous", False)
    if _is_hetero:
        for _t, _tdf in (ds.get("node_dfs") or {}).items():
            if body.column in _tdf.columns:
                ds["node_dfs"][_t], _ = impute_column(_tdf, body.column, body.method)
    # Refresh explore stats — rebuild per-type feature lists from stored schema spec.
    _node_type_features_imp: dict | None = None
    _edge_type_features_imp: dict | None = None
    if _is_hetero:
        _schema = ds.get("schema_spec") or {}
        _entries = _schema.get("entries", [])
        _ntf: dict[str, list[str]] = {}
        _etf: dict[str, list[str]] = {}
        for _e in _entries:
            if _e.get("xy") == "X" and _e.get("level") == "Node":
                _ntf.setdefault(_e["type"], []).append(_e["parameter"])
            elif _e.get("xy") == "X" and _e.get("level") == "Edge":
                _etf.setdefault(_e["type"], []).append(_e["parameter"])
        _node_type_features_imp = _ntf or None
        _edge_type_features_imp = _etf or None
    ds["explore_stats"] = compute_generic_explore(
        ds["nodes_df"], ds["edges_df"],
        is_heterogeneous=_is_hetero,
        node_types=ds.get("node_types", []),
        edge_types=ds.get("edge_types", []),
        canonical_edges=ds.get("canonical_edges", []),
        node_dfs=ds.get("node_dfs") if _is_hetero else None,
        edge_dfs=ds.get("edge_dfs") if _is_hetero else None,
        node_type_features=_node_type_features_imp,
        edge_type_features=_edge_type_features_imp,
    )
    store.put_dataset(ds["dataset_id"], ds)
    log = store.get_project(project_id).get("imputation_log", [])
    log.append({"column": body.column, "method": body.method, "filled_count": filled})
    store.update_project(project_id, imputation_log=log)
    return ImputationResult(column=body.column, filled_count=filled, method=body.method)


@router.post("/{project_id}/confirm", response_model=ProjectSummary)
async def confirm_data(project_id: str, body: ConfirmDataRequest):
    """Allow the user to override the Excel-declared task/label before training."""
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)

    df = ds.get("graph_df") if body.task_type.startswith("graph") else ds["nodes_df"]
    if df is None:
        df = ds["nodes_df"]
    result = validate_label(df, body.label_column, body.task_type)
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["message"])

    store.update_project(
        project_id,
        task_type=body.task_type, label_column=body.label_column,
        current_step=3, status="data_confirmed", updated_at=_now_iso(),
    )
    ds["task_type"] = body.task_type
    ds["label_column"] = body.label_column
    ds["num_classes"] = result.get("num_classes", 1) or 1
    store.put_dataset(ds["dataset_id"], ds)
    return _to_summary(store.get_project(project_id))


# ── Experiment hierarchy ──────────────────────────────────────────────────

def _to_experiment_summary(e: dict) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=e["experiment_id"], project_id=e["project_id"],
        name=e["name"], dataset_id=e["dataset_id"],
        task_type=e.get("task_type"), label_column=e.get("label_column"),
        current_step=e.get("current_step", 1),
        status=e.get("status", "created"),
        created_at=e["created_at"], updated_at=e.get("updated_at", e["created_at"]),
        run_count=len(e.get("task_ids", [])),
        best_metric=e.get("best_metric"), best_model=e.get("best_model"),
    )


@router.post("/{project_id}/experiments", response_model=ExperimentSummary)
async def create_experiment(project_id: str, body: CreateExperimentRequest):
    _project_or_404(project_id)
    ds = store.get_dataset(body.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    experiment_id = str(uuid.uuid4())
    now = _now_iso()
    record = {
        "experiment_id": experiment_id, "project_id": project_id,
        "name": body.name, "dataset_id": body.dataset_id,
        "task_type": None, "label_column": None,
        "current_step": 1, "status": "created",
        "created_at": now, "updated_at": now,
        "task_ids": [], "best_metric": None, "best_model": None,
    }
    store.put_experiment(experiment_id, record)
    project = store.get_project(project_id)
    exp_ids = project.get("experiment_ids", [])
    exp_ids.append(experiment_id)
    store.update_project(project_id, experiment_ids=exp_ids, updated_at=now)
    return _to_experiment_summary(record)


@router.get("/{project_id}/experiments/list", response_model=list[ExperimentSummary])
async def list_project_experiments(project_id: str):
    _project_or_404(project_id)
    return [_to_experiment_summary(e) for e in store.list_experiments(project_id)]


@router.get("/{project_id}/experiments/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(project_id: str, experiment_id: str):
    _project_or_404(project_id)
    exp = store.get_experiment(experiment_id)
    if not exp or exp["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Experiment not found")
    summary = _to_experiment_summary(exp)
    detail = ExperimentDetail(**summary.model_dump())
    ds = store.get_dataset(exp["dataset_id"])
    if ds:
        detail.dataset_summary = _dataset_to_summary(ds)
    runs = []
    for tid in exp.get("task_ids", []):
        task = store.get_task(tid)
        if task:
            runs.append(_task_to_status(task, project_id))
    detail.runs = runs
    return detail


@router.delete("/{project_id}/experiments/{experiment_id}")
async def delete_experiment(project_id: str, experiment_id: str):
    _project_or_404(project_id)
    exp = store.get_experiment(experiment_id)
    if not exp or exp["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Experiment not found")
    store.delete_experiment(experiment_id)
    project = store.get_project(project_id)
    exp_ids = project.get("experiment_ids", [])
    if experiment_id in exp_ids:
        exp_ids.remove(experiment_id)
        store.update_project(project_id, experiment_ids=exp_ids, updated_at=_now_iso())
    return {"detail": "Experiment deleted"}


# ── Training ───────────────────────────────────────────────────────────────

@router.get("/{project_id}/estimate", response_model=TrainingEstimate)
async def estimate_training_time(project_id: str, n_trials: int = Query(default=20)):
    project = _project_or_404(project_id)
    ds = _dataset_for_project(project)
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        cuda_ver = torch.version.cuda or "unknown"
        gpu_name = torch.cuda.get_device_name(0)
        device = f"cuda ({gpu_name}, CUDA {cuda_ver})"
    else:
        device = "cpu"
    num_nodes = ds["num_nodes"]
    history = store.get_training_history()
    if history:
        rates = []
        for h in history:
            if h.get("n_trials", 0) > 0 and h.get("num_nodes", 0) > 0:
                rate = h["duration_seconds"] / h["n_trials"] / (h["num_nodes"] / 1000)
                rates.append(rate)
        if rates:
            avg = sum(rates) / len(rates)
            return TrainingEstimate(estimated_seconds=round(avg * n_trials * (num_nodes / 1000), 1), device=device)
    seconds_per_trial = (2.0 + num_nodes / 10000 * 1.5) if has_cuda else (8.0 + num_nodes / 10000 * 6.0)
    return TrainingEstimate(estimated_seconds=round(seconds_per_trial * n_trials, 1), device=device)


@router.post("/{project_id}/train", response_model=TaskStatus)
async def start_training(
    project_id: str, body: StartTrainingRequest, background_tasks: BackgroundTasks,
):
    project = _project_or_404(project_id)
    if project.get("current_step", 1) < 3:
        raise HTTPException(status_code=400, detail="Data must be confirmed before training")
    task_id = str(uuid.uuid4())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = _now_iso()
    task_record = {
        "task_id": task_id, "project_id": project_id,
        "dataset_id": project["dataset_id"],
        "task_type": project["task_type"],
        "label_column": project["label_column"],
        "status": "QUEUED", "progress": 0,
        "current_trial": 0, "total_trials": body.n_trials, "device": device,
        "results": None, "report": None, "history": [], "error": None,
        "best_config": None,
        "models": body.models if body.models else None,
        "n_trials": body.n_trials,
        "started_at": started_at, "completed_at": None,
    }
    store.put_task(task_id, task_record)
    task_ids = project.get("task_ids", [])
    task_ids.append(task_id)
    store.update_project(
        project_id, task_id=task_id, task_ids=task_ids,
        training_config={"models": body.models, "n_trials": body.n_trials},
        current_step=3, status="training",
    )
    background_tasks.add_task(run_training_task, task_id)
    return TaskStatus(
        task_id=task_id, project_id=project_id, status="QUEUED",
        progress=0, current_trial=0, total_trials=body.n_trials,
        device=device, started_at=started_at,
    )


@router.get("/{project_id}/status", response_model=TaskStatus)
async def get_project_status(project_id: str):
    project = _project_or_404(project_id)
    task_id = project.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="No training task for this project")
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_status(task, project_id)


@router.get("/{project_id}/experiments", response_model=list[TaskStatus])
async def list_experiments(project_id: str):
    project = _project_or_404(project_id)
    results = []
    for tid in project.get("task_ids", []):
        task = store.get_task(tid)
        if task:
            results.append(_task_to_status(task, project_id))
    return results


@router.get("/{project_id}/report", response_model=Report)
async def get_project_report(project_id: str):
    project = _project_or_404(project_id)
    task_id = project.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="No training task for this project")
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Training not completed yet")
    report = task.get("report")
    if not report:
        raise HTTPException(status_code=404, detail="Report not available")
    return report


@router.get("/{project_id}/report/{task_id}", response_model=Report)
async def get_experiment_report(project_id: str, task_id: str):
    project = _project_or_404(project_id)
    if task_id not in project.get("task_ids", []):
        raise HTTPException(status_code=404, detail="Task not found for this project")
    task = store.get_task(task_id)
    if not task or task["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Training not completed yet")
    report = task.get("report")
    if not report:
        raise HTTPException(status_code=404, detail="Report not available")
    return report


# ── Model registry ─────────────────────────────────────────────────────────

@router.get("/{project_id}/models", response_model=list[RegisteredModel])
async def list_project_models(project_id: str):
    _project_or_404(project_id)
    return [RegisteredModel(**r) for r in store.list_model_records(project_id)]


@router.get("/{project_id}/models/{model_id}", response_model=RegisteredModel)
async def get_model_detail(project_id: str, model_id: str):
    _project_or_404(project_id)
    record = store.get_model_record(model_id)
    if not record or record.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Model not found")
    return RegisteredModel(**record)


@router.patch("/{project_id}/models/{model_id}", response_model=RegisteredModel)
async def update_model_info(project_id: str, model_id: str, body: RegisterModelRequest):
    _project_or_404(project_id)
    record = store.get_model_record(model_id)
    if not record or record.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Model not found")
    if body.name:
        record["name"] = body.name
    if body.description is not None:
        record["description"] = body.description
    store.put_model_record(model_id, record)
    return RegisteredModel(**record)


@router.delete("/{project_id}/models/{model_id}")
async def delete_model(project_id: str, model_id: str):
    _project_or_404(project_id)
    record = store.get_model_record(model_id)
    if not record or record.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Model not found")
    model_path = Path(record.get("file_path", ""))
    if model_path.exists():
        model_path.unlink()
    store.delete_model_record(model_id)
    return {"detail": "Model deleted"}
