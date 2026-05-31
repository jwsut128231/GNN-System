"""Training pipeline for Excel-ingested datasets (Phase 3 quality build).

Three clean branches based on the dataset record:
    1. graph + hetero  → list[HeteroData] + HeteroGraphRegressor (to_hetero)
    2. graph + homo    → list[Data]       + standard homo GNN (pool head)
    3. node            → single Data with train/test masks

Lightning best-practices applied across all three branches:
    * explicit accelerator + devices + precision (GPU-first)
    * ModelCheckpoint(monitor="val_loss", save_top_k=1) → best weights loaded
      before evaluation + serialization
    * gradient_clip_val
    * deterministic seed (seed_everything)
    * LR scheduler inside each model (ReduceLROnPlateau on val_loss)
    * EarlyStopping(monitor="val_loss", patience)
    * Regression target standardization (TargetScaler) — prevents the
      "R² = -99" pathology when target magnitudes are large.
    * 3-way split 60/20/20 for graph-level datasets of ≥10 graphs
      (falls back to 80/20 for smaller datasets, logged as warning).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
    confusion_matrix as sklearn_confusion_matrix,
)
from torch_geometric.loader import DataLoader

from app.core import store
from app.core.config import settings
from app.data.pyg_converter import dataframes_to_pyg_dynamic, dataframes_to_graph_list
from app.data.pyg_converter_hetero import parsed_excel_to_hetero_list
from app.models.factory import get_model
from app.training.callbacks import ProgressCallback
from app.training.optuna_search import run_hpo
from app.training.target_scaler import TargetScaler

log = logging.getLogger(__name__)


# ── metric helpers ────────────────────────────────────────────────────────

def _regression_metrics(y_true, y_pred) -> dict:
    arr = np.asarray(y_true)
    mape = None if (arr == 0).any() else round(float(mean_absolute_percentage_error(y_true, y_pred)), 4)
    return {
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2_score": round(float(r2_score(y_true, y_pred)), 4),
        "mape": mape,
    }


def _classification_metrics(y_true, y_pred) -> dict:
    avg = "binary" if len(set(y_true.tolist())) <= 2 else "macro"
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
    }


# ── device / Trainer helpers ─────────────────────────────────────────────

def _device_pair() -> tuple[str, str]:
    """Return (accelerator, precision) picked for the current host."""
    if torch.cuda.is_available():
        return "gpu", settings.PRECISION
    return "cpu", "32-true"


def _trainer(task_id: str, max_epochs: int, callbacks: list, checkpoint_dir: Path,
             accelerator: str, precision: str, is_regression: bool = False) -> pl.Trainer:
    monitor = "val_mae" if is_regression else "val_loss"
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir), filename="best",
        monitor=monitor, mode="min", save_top_k=1, save_weights_only=True,
    )
    es = pl.callbacks.EarlyStopping(
        monitor=monitor, patience=settings.PATIENCE, mode="min",
    )
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=settings.GRADIENT_CLIP,
        callbacks=[*callbacks, ckpt, es],
        enable_progress_bar=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        logger=False,
    ), ckpt


# ── split helpers ─────────────────────────────────────────────────────────

def _split_three(items: list, generator: torch.Generator) -> tuple[list, list, list]:
    """60/20/20 split with a fixed RNG; falls back to 80/20 for small lists."""
    n = len(items)
    if n < 5:
        perm = torch.randperm(n, generator=generator).tolist()
        split = max(int(n * 0.8), 1)
        tr = [items[i] for i in perm[:split]]
        te = [items[i] for i in perm[split:]] or tr[-1:]
        log.warning("3-way split skipped: only %d graphs; using 80/20 + val=test.", n)
        return tr, te, te
    perm = torch.randperm(n, generator=generator).tolist()
    t1 = int(n * 0.6)
    t2 = int(n * 0.8)
    return ([items[i] for i in perm[:t1]],
            [items[i] for i in perm[t1:t2]],
            [items[i] for i in perm[t2:]])


# ── data prep branches ────────────────────────────────────────────────────

def _label_columns_for(dataset: dict) -> list[str]:
    return list(dataset.get("label_columns") or [dataset["label_column"]])


def _prepare_hetero(dataset: dict, generator: torch.Generator):
    label_columns = _label_columns_for(dataset)
    parsed = {
        "node_dfs": dataset["node_dfs"],
        "edge_dfs": dataset["edge_dfs"],
        "graph_df": dataset["graph_df"],
        "label_column": dataset["label_column"],
        "label_columns": label_columns,
        "canonical_edges": dataset["canonical_edges"],
    }
    data_list, _s, _f, _ce, _excl = parsed_excel_to_hetero_list(parsed)
    metadata = data_list[0].metadata()
    num_classes = 1 if dataset["task_type"].endswith("regression") else 2
    train, val, test = _split_three(data_list, generator)

    # Regression target standardization (scalar OR per-target vector).
    is_regression = dataset["task_type"].endswith("regression")
    scaler = TargetScaler.identity_()
    if is_regression:
        train_y = np.concatenate([d.y.cpu().numpy() for d in train], axis=0)
        scaler = TargetScaler.fit(train_y)
        for d in train + val:   # apply only to train+val; test stays raw for unscaled metric
            d.y = scaler.transform_tensor(d.y)
    return train, val, test, metadata, num_classes, scaler


def _prepare_graph_homo(dataset: dict, generator: torch.Generator):
    label_columns = _label_columns_for(dataset)
    data_list, _s, _f, num_classes = dataframes_to_graph_list(
        dataset["nodes_df"], dataset["edges_df"], dataset.get("graph_df"),
        label_column=label_columns, task_type=dataset["task_type"],
        fit_scaler=True,
    )
    train, val, test = _split_three(data_list, generator)

    is_regression = dataset["task_type"].endswith("regression")
    scaler = TargetScaler.identity_()
    if is_regression:
        train_y = np.concatenate([d.y.cpu().numpy() for d in train], axis=0)
        scaler = TargetScaler.fit(train_y)
        for d in train + val:
            d.y = scaler.transform_tensor(d.y)
    return train, val, test, num_classes, scaler


def _prepare_node(dataset: dict):
    label_columns = _label_columns_for(dataset)
    train_data, _scaler, _ = dataframes_to_pyg_dynamic(
        dataset["nodes_df_train"], dataset["edges_df_train"],
        label_column=label_columns, task_type=dataset["task_type"],
        fit_scaler=True,
    )
    test_data, _, _ = dataframes_to_pyg_dynamic(
        dataset["nodes_df_test"], dataset["edges_df_test"],
        label_column=label_columns, task_type=dataset["task_type"],
        fit_scaler=True,
    )
    num_classes = getattr(train_data, "num_classes", 2)

    scaler = TargetScaler.identity_()
    if dataset["task_type"].endswith("regression"):
        scaler = TargetScaler.fit(train_data.y.cpu().numpy())
        train_data.y = scaler.transform_tensor(train_data.y)
    return train_data, test_data, num_classes, scaler


# ── prediction helpers ────────────────────────────────────────────────────

def _predict_list(model, data_list, task_type: str, is_hetero: bool,
                  scaler: TargetScaler) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(data_list, batch_size=max(len(data_list), 1))
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in loader:
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

            out_np = out.detach().cpu().numpy()
            y_np = batch.y.cpu().numpy()
            if task_type.endswith("regression"):
                # NOTE: test items kept unscaled in y; model output is scaled → unscale
                y_pred_all.append(scaler.inverse_np(out_np))
                y_true_all.append(y_np)
            else:
                y_pred_all.append(out.argmax(dim=-1).detach().cpu().numpy())
                y_true_all.append(y_np)
    return np.concatenate(y_pred_all), np.concatenate(y_true_all)


def _predict_single(model, data, task_type: str, scaler: TargetScaler):
    model.eval()
    with torch.no_grad():
        data = data.to(model.device)
        out = model(data.x, data.edge_index,
                    getattr(data, "edge_attr", None), batch=None)
    y_true = data.y.cpu().numpy()
    out_np = out.cpu().numpy()
    if task_type.endswith("regression"):
        # For node-level path the test y was not scaled; scale-invert preds only.
        return scaler.inverse_np(out_np), y_true
    return out.argmax(dim=-1).cpu().numpy(), y_true


# ── main entry point ──────────────────────────────────────────────────────

def run_training_task(task_id: str) -> None:
    try:
        pl.seed_everything(settings.DETERMINISTIC_SEED, workers=True)

        task = store.get_task(task_id)
        dataset = store.get_dataset(task["dataset_id"])
        project_id = task.get("project_id")
        task_type = task.get("task_type", dataset.get("task_type"))
        n_trials = task.get("n_trials", settings.OPTUNA_TRIALS)
        models_filter = task.get("models")

        # Multi-Y metadata.
        label_columns: list[str] = list(
            dataset.get("label_columns") or [dataset.get("label_column")]
        )
        label_weights_list: list[float] = list(
            dataset.get("label_weights") or [dataset.get("label_weight") or 1.0]
        )
        num_targets = len(label_columns)
        loss_weights_tensor = (
            torch.tensor(label_weights_list, dtype=torch.float)
            if num_targets > 1 else None
        )

        accelerator, precision = _device_pair()
        device_str = "cuda" if accelerator == "gpu" else "cpu"
        if accelerator == "gpu":
            cuda_ver = torch.version.cuda or "unknown"
            gpu_name = torch.cuda.get_device_name(0)
            device_str = f"cuda ({gpu_name}, CUDA {cuda_ver})"

        store.update_task(
            task_id, device=device_str, status="PREPROCESSING", progress=5,
            current_phase="preprocessing",
            current_trial=0, total_trials=n_trials,
        )

        is_hetero = bool(dataset.get("is_heterogeneous"))
        is_graph_task = task_type.startswith("graph")
        gen = torch.Generator().manual_seed(settings.DETERMINISTIC_SEED)

        # ── Prepare data ──
        metadata = None
        scaler = TargetScaler.identity_()
        if is_graph_task and is_hetero:
            train_items, val_items, test_items, metadata, num_classes, scaler = \
                _prepare_hetero(dataset, gen)
        elif is_graph_task:
            train_items, val_items, test_items, num_classes, scaler = \
                _prepare_graph_homo(dataset, gen)
        else:
            train_single, test_single, num_classes, scaler = _prepare_node(dataset)
            train_items, val_items, test_items = train_single, test_single, test_single

        # num_features for homogeneous paths
        if isinstance(train_items, list):
            sample = train_items[0]
            num_features = int(next(iter(sample.x_dict.values())).shape[1]) if is_hetero \
                else int(sample.x.shape[1])
        else:
            num_features = int(train_items.x.shape[1])

        store.update_task(
            task_id, progress=15, status="TRAINING",
            current_phase="hpo",
            current_trial=0, total_trials=n_trials,
        )

        # ── HPO: unified across node / graph-homo / graph-hetero ──
        best_config = run_hpo(
            train_items=train_items, val_items=val_items,
            num_features=num_features, n_trials=n_trials,
            task_type=task_type, models=models_filter, task_id=task_id,
            accelerator=accelerator, precision=precision,
            metadata=metadata,
            num_targets=num_targets,
            loss_weights=loss_weights_tensor,
        )

        store.update_task(
            task_id, progress=50, best_config=best_config,
            current_phase="final_training",
            current_trial=best_config.get("completed_trials", n_trials),
            total_trials=n_trials,
        )

        # ── Build model ──
        is_regression = task_type.endswith("regression")
        effective_classes = 1 if is_regression else num_classes
        final_kwargs: dict = dict(
            num_features=num_features,
            num_classes=effective_classes,
            task_type=task_type,
            metadata=metadata,
            hidden_dim=best_config["hidden_dim"],
            num_layers=best_config["num_layers"],
            dropout=best_config["dropout"],
            lr=best_config["lr"],
        )
        if is_regression:
            final_kwargs["num_targets"] = num_targets
            if loss_weights_tensor is not None:
                final_kwargs["loss_weights"] = loss_weights_tensor
        model = get_model(best_config["model_name"], **final_kwargs)

        # ── DataLoaders ──
        if isinstance(train_items, list):
            batch_size = min(8, len(train_items)) or 1
            train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_items, batch_size=batch_size)
        else:
            train_loader = DataLoader([train_items], batch_size=1, shuffle=False)
            val_loader = DataLoader([val_items], batch_size=1, shuffle=False)

        ckpt_dir = Path(settings.STORAGE_DIR) / "checkpoints" / task_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Effective max_epochs bound for progress mapping:
        # early stop can cut training short, so cap the denominator at
        # (PATIENCE * 2 + 5) for nicer progress pacing on plateaus.
        effective_max = min(settings.MAX_EPOCHS, settings.PATIENCE * 4 + 5)
        progress_cb = ProgressCallback(
            task_id=task_id, max_epochs=effective_max,
            phase_range=(50, 99), task_type=task_type,
        )
        trainer, ckpt_cb = _trainer(
            task_id=task_id, max_epochs=settings.MAX_EPOCHS,
            callbacks=[progress_cb], checkpoint_dir=ckpt_dir,
            accelerator=accelerator, precision=precision,
            is_regression=is_regression,
        )

        t0 = time.time()
        trainer.fit(model, train_loader, val_loader)
        train_time = time.time() - t0

        # ── Load best checkpoint for evaluation ──
        best_path = ckpt_cb.best_model_path
        if best_path and Path(best_path).exists():
            state = torch.load(best_path, map_location=model.device)
            model.load_state_dict(state["state_dict"])

        # ── Evaluation ──
        # Graph-level paths (_prepare_hetero / _prepare_graph_homo) scale BOTH
        # train and val items, so train_y AND val_y must be inverse-scaled
        # before metric computation. Forgetting val_y produced absurd val
        # metrics (R²=-30+, MAPE in the thousands) — see 2026-04-28 fix.
        # Node-level path (_prepare_node) scales train only; val_items reuses
        # the unscaled test data, so val_y is already in raw space and must
        # NOT be inverse-scaled a second time.
        if isinstance(train_items, list):
            train_preds, train_y = _predict_list(model, train_items, task_type, is_hetero, scaler)
            val_preds, val_y = _predict_list(model, val_items, task_type, is_hetero, scaler)
            test_preds, test_y = _predict_list(model, test_items, task_type, is_hetero, scaler)
            if is_regression:
                train_y = scaler.inverse_np(train_y)
                val_y = scaler.inverse_np(val_y)
        else:
            train_preds, train_y = _predict_single(model, train_items, task_type, scaler)
            val_preds, val_y = _predict_single(model, val_items, task_type, scaler)
            test_preds, test_y = _predict_single(model, test_items, task_type, scaler)
            if is_regression:
                train_y = scaler.inverse_np(train_y)

        per_target_metrics: dict = {}
        per_target_residuals: dict = {}
        if is_regression:
            if num_targets > 1:
                # Compute metrics per Y column; aggregate by mean for the
                # backward-compatible overall train/val/test metrics fields.
                target_metrics_test: list[dict] = []
                target_metrics_train: list[dict] = []
                target_metrics_val: list[dict] = []
                for i, col in enumerate(label_columns):
                    tr_m = _regression_metrics(train_y[:, i], train_preds[:, i])
                    va_m = _regression_metrics(val_y[:, i], val_preds[:, i])
                    te_m = _regression_metrics(test_y[:, i], test_preds[:, i])
                    target_metrics_train.append(tr_m)
                    target_metrics_val.append(va_m)
                    target_metrics_test.append(te_m)
                    per_target_metrics[col] = te_m
                    per_target_residuals[col] = [
                        {
                            "actual": round(float(test_y[j, i]), 4),
                            "predicted": round(float(test_preds[j, i]), 4),
                            "error": round(float(test_y[j, i] - test_preds[j, i]), 4),
                        }
                        for j in range(min(500, len(test_y)))
                    ]
                train_metrics = {
                    k: round(float(np.mean([m[k] for m in target_metrics_train])), 4)
                    for k in target_metrics_train[0]
                }
                val_metrics = {
                    k: round(float(np.mean([m[k] for m in target_metrics_val])), 4)
                    for k in target_metrics_val[0]
                }
                test_metrics = {
                    k: round(float(np.mean([m[k] for m in target_metrics_test])), 4)
                    for k in target_metrics_test[0]
                }
                cm = None
                # Residual scatter falls back to the first target for the
                # legacy single-axis plot; per-target plots live in
                # ``per_target_residuals``.
                residual = per_target_residuals[label_columns[0]]
            else:
                train_metrics = _regression_metrics(train_y, train_preds)
                val_metrics = _regression_metrics(val_y, val_preds)
                test_metrics = _regression_metrics(test_y, test_preds)
                cm = None
                residual = [
                    {
                        "actual": round(float(test_y[i]), 4),
                        "predicted": round(float(test_preds[i]), 4),
                        "error": round(float(test_y[i] - test_preds[i]), 4),
                    }
                    for i in range(min(500, len(test_y)))
                ]
        else:
            train_metrics = _classification_metrics(train_y, train_preds)
            val_metrics = _classification_metrics(val_y, val_preds)
            test_metrics = _classification_metrics(test_y, test_preds)
            labels = sorted(set(test_y.tolist()) | set(test_preds.tolist()))
            cm_arr = sklearn_confusion_matrix(test_y, test_preds, labels=labels)
            cm = {"labels": [str(l) for l in labels], "matrix": cm_arr.tolist()}
            residual = None

        report = {
            "task_type": task_type,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": progress_cb.history,
            "confusion_matrix": cm,
            "residual_data": residual,
            "node_predictions": [],
            "best_config": {
                "model_name": best_config["model_name"],
                "hidden_dim": best_config["hidden_dim"],
                "num_layers": best_config["num_layers"],
                "dropout": round(best_config["dropout"], 3),
                "lr": round(best_config["lr"], 6),
            },
            "leaderboard": best_config.get("leaderboard", []),
            "is_heterogeneous": is_hetero,
            "label_columns": label_columns,
            "per_target_metrics": per_target_metrics,
            "per_target_residuals": per_target_residuals,
        }

        store.update_task(
            task_id, status="COMPLETED", progress=100,
            current_phase="completed",
            results={
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "training_time_seconds": round(train_time, 1),
            },
            best_config=report["best_config"], report=report,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Persist to model registry
        models_dir = Path(settings.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_file = models_dir / f"{task_id}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "model_name": best_config["model_name"],
            "num_features": num_features,
            "num_classes": effective_classes,
            "task_type": task_type,
            "label_column": dataset.get("label_column"),
            "label_columns": label_columns,
            "label_weights": label_weights_list,
            "num_targets": num_targets,
            "hidden_dim": best_config["hidden_dim"],
            "num_layers": best_config["num_layers"],
            "dropout": best_config["dropout"],
            "lr": best_config["lr"],
            "is_heterogeneous": is_hetero,
            "metadata": metadata,
            "target_scaler": scaler.to_dict(),
        }, str(model_file))

        store.put_model_record(task_id, {
            "model_id": task_id, "project_id": project_id or "",
            "task_id": task_id,
            "name": f"{best_config['model_name'].upper()} - {task_type}",
            "model_name": best_config["model_name"],
            "task_type": task_type,
            "label_column": dataset.get("label_column"),
            "num_features": num_features,
            "num_classes": effective_classes,
            "best_config": report["best_config"],
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "file_path": str(model_file),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "description": "",
        })

        store.add_training_record({
            "num_nodes": dataset.get("num_nodes", 0),
            "n_trials": n_trials,
            "duration_seconds": round(train_time, 1),
        })

        if project_id:
            store.update_project(project_id, current_step=4, status="completed")

    except Exception:
        log.exception("Training task %s failed", task_id)
        store.update_task(task_id, status="FAILED", progress=0,
                          current_phase="failed",
                          error="Training failed. Check server logs for details.")
        tk = store.get_task(task_id) or {}
        if tk.get("project_id"):
            store.update_project(tk["project_id"], status="failed")
