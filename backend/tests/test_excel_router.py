"""Router-level smoke tests for the Excel upload endpoint."""
from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import store
from app.data import graph_cache_sqlite
from app.main import app


@pytest.fixture(autouse=True)
def clean_store():
    store.datasets.clear()
    store.projects.clear()
    store.tasks.clear()
    yield
    store.datasets.clear()
    store.projects.clear()
    store.tasks.clear()


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _valid_excel_bytes() -> bytes:
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label", "Weight": None},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1] * 6,
        "Node": [0, 1, 2, 3, 4, 5],
        "X_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "label": [0, 1, 0, 1, 0, 1],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        parameter.to_excel(w, sheet_name="Parameter", index=False)
        nodes.to_excel(w, sheet_name="Node", index=False)
    buf.seek(0)
    return buf.read()


def _create_project(client) -> str:
    resp = client.post("/api/v1/projects/", json={"name": "excel-test", "tags": []})
    assert resp.status_code == 200, resp.text
    return resp.json()["project_id"]


def test_sample_excel_download(client):
    resp = client.get("/api/v1/projects/sample-excel")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert len(resp.content) > 0


def test_upload_excel_success(client):
    project_id = _create_project(client)
    files = {
        "file": ("template.xlsx", _valid_excel_bytes(),
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    }
    resp = client.post(
        f"/api/v1/projects/{project_id}/upload-excel",
        files=files,
        data={"dataset_name": "my-data"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["declared_task_type"] == "node_classification"
    assert body["declared_label_column"] == "label"
    assert body["num_nodes"] == 6
    assert body["schema_spec"]["entries"][1]["xy"] == "Y"

    # Project should have advanced to step 3 (data_confirmed) automatically.
    proj = client.get(f"/api/v1/projects/{project_id}").json()
    assert proj["current_step"] == 3
    assert proj["task_type"] == "node_classification"
    assert proj["label_column"] == "label"


def test_upload_excel_invalid_file(client):
    project_id = _create_project(client)
    files = {
        "file": ("bogus.xlsx", b"not an excel file",
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    }
    resp = client.post(
        f"/api/v1/projects/{project_id}/upload-excel", files=files,
    )
    assert resp.status_code == 422
    assert "read Excel file" in resp.json()["detail"]


def test_upload_excel_unknown_project(client):
    files = {
        "file": ("template.xlsx", _valid_excel_bytes(),
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    }
    resp = client.post("/api/v1/projects/does-not-exist/upload-excel", files=files)
    assert resp.status_code == 404


# ── Integration tests: graph_index, ETag, cache invalidation ──────────────

@pytest.fixture(autouse=True)
def reset_sqlite_cache():
    graph_cache_sqlite.reset_for_test()
    yield
    graph_cache_sqlite.reset_for_test()


def _upload_excel(client, project_id: str, content: bytes) -> None:
    files = {
        "file": ("template.xlsx", content,
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    }
    resp = client.post(
        f"/api/v1/projects/{project_id}/upload-excel",
        files=files,
        data={"dataset_name": "test-ds"},
    )
    assert resp.status_code == 200, resp.text


def _valid_excel_bytes_v2() -> bytes:
    """Slightly different content (X_1 values changed) to force cache invalidation."""
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "X_1", "Weight": None},
        {"XY": "Y", "Level": "Node", "Type": "default", "Parameter": "label", "Weight": None},
    ])
    nodes = pd.DataFrame({
        "Graph_ID": [1] * 6,
        "Node": [0, 1, 2, 3, 4, 5],
        "X_1": [9.9, 8.8, 7.7, 6.6, 5.5, 4.4],
        "label": [1, 0, 1, 0, 1, 0],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        parameter.to_excel(w, sheet_name="Parameter", index=False)
        nodes.to_excel(w, sheet_name="Node", index=False)
    buf.seek(0)
    return buf.read()


def test_graph_sample_returns_graph_index_in_response(client):
    project_id = _create_project(client)
    _upload_excel(client, project_id, _valid_excel_bytes())

    resp = client.get(f"/api/v1/projects/{project_id}/graph-sample")
    assert resp.status_code == 200
    body = resp.json()
    assert "graph_index" in body
    assert isinstance(body["graph_index"], list)
    assert len(body["graph_index"]) >= 1
    entry = body["graph_index"][0]
    assert "id" in entry
    assert "node_count" in entry
    assert "edge_count" in entry


def test_graph_sample_etag_304(client):
    project_id = _create_project(client)
    _upload_excel(client, project_id, _valid_excel_bytes())

    # First request — should return 200 with ETag
    resp1 = client.get(f"/api/v1/projects/{project_id}/graph-sample")
    assert resp1.status_code == 200
    etag = resp1.headers.get("etag")
    assert etag is not None

    # Second request with If-None-Match → 304
    resp2 = client.get(
        f"/api/v1/projects/{project_id}/graph-sample",
        headers={"if-none-match": etag},
    )
    assert resp2.status_code == 304


def test_upload_reupload_same_project_diff_content_invalidates_cache(client):
    project_id = _create_project(client)
    v1_bytes = _valid_excel_bytes()
    _upload_excel(client, project_id, v1_bytes)

    # Grab dataset_id and seed the SQLite cache manually
    project = store.get_project(project_id)
    ds_id = project["dataset_id"]
    ds = store.get_dataset(ds_id)
    old_hash = ds.get("excel_hash", "")
    graph_cache_sqlite.put(ds_id, "1", old_hash, b"stale-payload")
    assert graph_cache_sqlite.get(ds_id, "1", old_hash) == b"stale-payload"

    # Re-upload with different content
    _upload_excel(client, project_id, _valid_excel_bytes_v2())

    # Old cache entry must have been invalidated
    assert graph_cache_sqlite.get(ds_id, "1", old_hash) is None
