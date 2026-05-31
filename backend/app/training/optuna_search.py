"""Optuna hyperparameter search with per-trial early stopping.

Supports three training kinds:
    * node-level  — single ``Data`` for train/val
    * graph homo  — ``list[Data]`` for train/val
    * graph hetero— ``list[HeteroData]`` for train/val (requires ``metadata``)

Every trial trains a fresh model for ``MAX_HPO_EPOCHS`` with an
``EarlyStopping`` callback on ``val_loss``. The Optuna study itself uses a
``MedianPruner``. A ``TrialProgressCallback`` pushes trial progress back to
the task store so the UI can display ``Trial X / N`` live.
"""
from __future__ import annotations

from typing import Optional, Union

import optuna
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

from app.core.config import settings
from app.models.factory import (
    get_model,
    HOMO_REGISTRY as MODEL_REGISTRY,
    HETERO_BACKBONES,
)
from app.training.callbacks import TrialProgressCallback


TrainItems = Union[Data, list[Data], list[HeteroData]]
Metadata = tuple[list[str], list[tuple[str, str, str]]]


def _trainer_kwargs(accelerator: str, precision: str) -> dict:
    return {
        "max_epochs": settings.MAX_HPO_EPOCHS,
        "accelerator": accelerator,
        "devices": 1,
        "precision": precision,
        "gradient_clip_val": settings.GRADIENT_CLIP,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "enable_checkpointing": False,
        "logger": False,
    }


def _build_loaders(train_items: TrainItems, val_items: TrainItems) \
        -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders appropriate for the input shape."""
    if isinstance(train_items, list):
        batch_size = min(8, len(train_items)) or 1
        train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_items if isinstance(val_items, list) else [val_items],
            batch_size=batch_size,
        )
    else:
        train_loader = DataLoader([train_items], batch_size=1, shuffle=False)
        val_loader = DataLoader([val_items], batch_size=1, shuffle=False)
    return train_loader, val_loader


def run_hpo(
    train_data: TrainItems = None,
    val_data: TrainItems = None,
    num_features: int = 0,
    n_trials: int = 20,
    class_weights: torch.Tensor | None = None,
    task_type: str = "node_classification",
    models: Optional[list[str]] = None,
    task_id: Optional[str] = None,
    accelerator: str = "auto",
    precision: str = "32-true",
    metadata: Optional[Metadata] = None,
    num_targets: int = 1,
    loss_weights: Optional[torch.Tensor] = None,
    *,
    train_items: Optional[TrainItems] = None,
    val_items: Optional[TrainItems] = None,
) -> dict:
    """Run Optuna HPO. Returns the best hyperparameter dict + leaderboard.

    ``train_data`` / ``val_data`` are preserved for backwards compatibility
    with the node-level caller. ``train_items`` / ``val_items`` are the new
    kwarg names that make the graph-level path read naturally.
    """
    train_items = train_items if train_items is not None else train_data
    val_items = val_items if val_items is not None else val_data
    if train_items is None or val_items is None:
        raise ValueError("run_hpo requires train_items (or train_data) and val_items (or val_data).")

    is_regression = task_type.endswith("regression")
    is_hetero = metadata is not None
    num_classes = 1 if is_regression else 2

    if is_hetero:
        available_models = list(HETERO_BACKBONES)
    else:
        available_models = list(MODEL_REGISTRY.keys())
    if models:
        search_models = [m for m in models if m in available_models] or available_models
    else:
        search_models = available_models

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", search_models)
        # Tightened search ranges (2026-04-28): the original (1e-4, 1e-2) lr
        # range produced training loss that bounced around; constrained to
        # 1e-5..1e-4 paired with the bumped MAX_EPOCHS=200 / PATIENCE=30
        # budget gives stable, monotonically-decreasing val curves on
        # demo-scale data. Dropout capped at 0.3 (>0.3 starves signal on
        # small graphs). hidden_dim 256 + num_layers 5 dropped — rarely
        # selected and increase over-smoothing risk on tiny graphs.
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)

        model_kwargs = dict(
            num_features=num_features, num_classes=num_classes,
            hidden_dim=hidden_dim, num_layers=num_layers,
            dropout=dropout, lr=lr, task_type=task_type,
        )
        if not is_regression and not is_hetero:
            model_kwargs["class_weights"] = class_weights
        if is_hetero:
            model_kwargs["metadata"] = metadata
        # Multi-Y regression: forward num_targets + loss_weights to all paths.
        if is_regression:
            model_kwargs["num_targets"] = num_targets
            if loss_weights is not None:
                model_kwargs["loss_weights"] = loss_weights

        model = get_model(model_name, **model_kwargs)

        train_loader, val_loader = _build_loaders(train_items, val_items)
        early_stop = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=settings.HPO_PATIENCE, mode="min",
        )
        trainer = pl.Trainer(
            callbacks=[early_stop], **_trainer_kwargs(accelerator, precision),
        )
        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf")))
        return float(val_loss)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(5, max(1, n_trials // 4)))
    study = optuna.create_study(direction="minimize", pruner=pruner)

    callbacks = []
    if task_id:
        callbacks.append(TrialProgressCallback(task_id, n_trials=n_trials))

    study.optimize(objective, n_trials=n_trials, timeout=None, callbacks=callbacks)

    best = study.best_params
    leaderboard = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            leaderboard.append({
                "trial": trial.number,
                "model": trial.params.get("model", "unknown"),
                "hidden_dim": trial.params.get("hidden_dim", 0),
                "num_layers": trial.params.get("num_layers", 0),
                "dropout": round(trial.params.get("dropout", 0.0), 3),
                "lr": round(trial.params.get("lr", 0.0), 6),
                "val_loss": round(trial.value, 4) if trial.value is not None else float("inf"),
            })
    leaderboard.sort(key=lambda x: x["val_loss"])

    return {
        "model_name": best["model"],
        "hidden_dim": best["hidden_dim"],
        "num_layers": best["num_layers"],
        "dropout": best["dropout"],
        "lr": best["lr"],
        "leaderboard": leaderboard[:10],
        "completed_trials": len(leaderboard),
    }
