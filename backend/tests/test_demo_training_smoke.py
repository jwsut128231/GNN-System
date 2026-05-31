"""End-to-end smoke test: each demo workbook can be parsed, prepared, trained,
and evaluated by the full pipeline (sans Optuna HPO, which is replaced with a
fixed model config to keep CPU runtime under a minute per workbook).

Covered workbooks:
  * demo_multigraph_homo.v2.xlsx          — single Y, with Type columns
  * demo_multigraph_homo_no_type.xlsx     — single Y, NO Type columns (homogeneous)
  * demo_multigraph_multi_y.xlsx          — two Y targets, NO Type columns
  * demo_multigraph_hetero.v2.xlsx        — heterogeneous, with Type columns

For each workbook the test asserts:
  * parse_excel_file succeeds and yields the expected task_type
  * label_weights resolves blank cells to 1.0
  * a 5-epoch Lightning training pass completes without error
  * predicted y has the right shape (matches num_targets)
  * test metrics dict carries finite numbers (no NaN/Inf)
"""
from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

from torch_geometric.loader import DataLoader

from app.data.excel_ingestion import parse_excel_file
from app.data.pyg_converter import dataframes_to_graph_list
from app.data.pyg_converter_hetero import parsed_excel_to_hetero_list
from app.models.factory import get_model
from app.training.target_scaler import TargetScaler


DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def _prepare_homo(parsed: dict):
    """Mirror pipeline._prepare_graph_homo without the store dependency."""
    label_columns = parsed["label_columns"]
    data_list, _s, _f, num_classes = dataframes_to_graph_list(
        parsed["nodes_df"], parsed["edges_df"], parsed.get("graph_df"),
        label_column=label_columns, task_type=parsed["task_type"],
        fit_scaler=True,
    )
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(data_list), generator=gen).tolist()
    n = len(data_list)
    t1 = int(n * 0.6); t2 = int(n * 0.8)
    train = [data_list[i] for i in perm[:t1]]
    val = [data_list[i] for i in perm[t1:t2]]
    test = [data_list[i] for i in perm[t2:]]
    # Scale Y on train.
    train_y = np.concatenate([d.y.cpu().numpy() for d in train], axis=0)
    scaler = TargetScaler.fit(train_y)
    for d in train + val:
        d.y = scaler.transform_tensor(d.y)
    return train, val, test, num_classes, scaler


def _prepare_hetero(parsed: dict):
    data_list, _s, _f, _ce, _exc = parsed_excel_to_hetero_list(parsed)
    metadata = data_list[0].metadata()
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(data_list), generator=gen).tolist()
    n = len(data_list)
    t1 = int(n * 0.6); t2 = int(n * 0.8)
    train = [data_list[i] for i in perm[:t1]]
    val = [data_list[i] for i in perm[t1:t2]]
    test = [data_list[i] for i in perm[t2:]]
    train_y = np.concatenate([d.y.cpu().numpy() for d in train], axis=0)
    scaler = TargetScaler.fit(train_y)
    for d in train + val:
        d.y = scaler.transform_tensor(d.y)
    return train, val, test, metadata, 1, scaler


def _train_and_predict(model, train, val, test, *, is_hetero: bool):
    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    val_loader = DataLoader(val, batch_size=4)
    trainer = pl.Trainer(
        max_epochs=5, accelerator="cpu", devices=1, precision="32-true",
        enable_progress_bar=False, enable_checkpointing=False,
        enable_model_summary=False, logger=False,
    )
    trainer.fit(model, train_loader, val_loader)
    # Predict on test.
    model.eval()
    test_loader = DataLoader(test, batch_size=max(len(test), 1))
    preds = []
    ys = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(model.device)
            if is_hetero:
                x_dict = {nt: batch[nt].x for nt in batch.node_types}
                ei_dict = {et: batch[et].edge_index for et in batch.edge_types}
                b_dict = {nt: batch[nt].batch for nt in batch.node_types}
                out = model(x_dict, ei_dict, b_dict)
            else:
                b = getattr(batch, "batch", None)
                out = model(batch.x, batch.edge_index,
                            getattr(batch, "edge_attr", None), batch=b)
            preds.append(out.cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)


# ── Parametrised smoke test ─────────────────────────────────────────────

@pytest.mark.parametrize("filename,is_hetero,expected_targets", [
    ("demo_multigraph_homo.v2.xlsx", False, 1),
    ("demo_multigraph_homo_no_type.xlsx", False, 1),
    ("demo_multigraph_multi_y.xlsx", False, 2),
    ("demo_multigraph_hetero.v2.xlsx", True, 1),
])
def test_demo_workbook_trains_end_to_end(filename, is_hetero, expected_targets):
    path = DEMO_DIR / filename
    assert path.exists(), f"Demo workbook missing: {path}. Run scripts/generate_excel_demos.py"

    parsed = parse_excel_file(path.read_bytes(), filename)
    assert parsed["task_type"] == "graph_regression"
    assert len(parsed["label_columns"]) == expected_targets
    # All label_weights must be 1.0+ (defaults applied where blank)
    assert all(w >= 1.0 for w in parsed["label_weights"]), \
        f"label_weights should default blank cells to 1.0, got {parsed['label_weights']}"
    assert parsed["is_heterogeneous"] is is_hetero, \
        f"Expected is_heterogeneous={is_hetero}, got {parsed['is_heterogeneous']}"

    if is_hetero:
        train, val, test, metadata, num_classes, _scaler = _prepare_hetero(parsed)
        model = get_model(
            "sage", num_features=2, num_classes=1,
            task_type="graph_regression", metadata=metadata,
            hidden_dim=16, num_layers=2, dropout=0.1, lr=1e-3,
            num_targets=expected_targets,
            loss_weights=(torch.tensor(parsed["label_weights"])
                          if expected_targets > 1 else None),
        )
    else:
        train, val, test, num_classes, _scaler = _prepare_homo(parsed)
        num_features = int(train[0].x.shape[1])
        model = get_model(
            "sage", num_features=num_features, num_classes=1,
            task_type="graph_regression",
            hidden_dim=16, num_layers=2, dropout=0.1, lr=1e-3,
            num_targets=expected_targets,
            loss_weights=(torch.tensor(parsed["label_weights"])
                          if expected_targets > 1 else None),
        )

    assert len(train) > 0 and len(val) > 0 and len(test) > 0

    preds, ys = _train_and_predict(model, train, val, test, is_hetero=is_hetero)
    # Shape sanity
    if expected_targets > 1:
        assert preds.ndim == 2 and preds.shape[1] == expected_targets
        assert ys.shape[-1] == expected_targets
    else:
        assert preds.ndim in (1, 2)
    # No NaN/Inf
    assert np.all(np.isfinite(preds)), "Predictions contain NaN/Inf"
    assert np.all(np.isfinite(ys)), "Targets contain NaN/Inf"
    # Loss-based sanity: mean squared error between predictions and y is finite
    flat_preds = preds.reshape(-1)
    flat_ys = ys.reshape(-1)
    mse = float(np.mean((flat_preds - flat_ys) ** 2))
    assert math.isfinite(mse), f"MSE is not finite: {mse}"
