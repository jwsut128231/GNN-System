"""Lightning callbacks used by the training pipeline.

Two callbacks are provided:

``ProgressCallback``
    Phase-aware progress reporter. Given a ``(start_pct, end_pct)`` window and
    an ``effective_max_epochs`` (typically ``min(max_epochs, es_patience_budget)``),
    it writes a monotonic, bounded progress value to the task store after every
    train+validation epoch and appends a history entry with current LR.

``TrialProgressCallback``
    Lightweight Optuna-trial progress reporter. Maps completed trials to a
    fraction of the HPO phase window.
"""
from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl

from app.core import store


class ProgressCallback(pl.Callback):
    """Updates the task store once per validation epoch.

    Progress is mapped into a caller-supplied phase window so the outer
    pipeline can compose phases (PREPROCESSING → HPO → TRAINING → DONE) into
    one monotonic 0–100 bar.
    """

    def __init__(
        self,
        task_id: str,
        max_epochs: int,
        phase_range: tuple[int, int] = (50, 99),
        task_type: str = "graph_regression",
    ) -> None:
        self.task_id = task_id
        self.max_epochs = max(int(max_epochs), 1)
        self.phase_start, self.phase_end = phase_range
        self.task_type = task_type
        self.history: list[dict] = []
        self._last_progress = self.phase_start

    # ── helpers ──

    def _phase_progress(self, epoch_idx_1based: int) -> int:
        frac = min(epoch_idx_1based / self.max_epochs, 1.0)
        raw = self.phase_start + int(round((self.phase_end - self.phase_start) * frac))
        # enforce monotonicity
        raw = max(raw, self._last_progress)
        raw = min(raw, self.phase_end)
        self._last_progress = raw
        return raw

    def _current_lr(self, trainer: pl.Trainer) -> Optional[float]:
        try:
            opt = trainer.optimizers[0]
            return float(opt.param_groups[0]["lr"])
        except (IndexError, AttributeError, KeyError):
            return None

    # ── Lightning hook ──

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_1 = int(trainer.current_epoch) + 1
        metrics = trainer.callback_metrics
        train_loss = float(metrics.get("train_loss", metrics.get("train_loss_epoch", 0.0)))
        val_loss = float(metrics.get("val_loss", metrics.get("val_loss_epoch", 0.0)))
        lr = self._current_lr(trainer)

        is_classification = not self.task_type.endswith("regression")
        acc = None
        if is_classification:
            acc_val = metrics.get("val_acc", metrics.get("train_acc"))
            if acc_val is not None:
                acc = round(float(acc_val), 4)

        entry = {
            "epoch": epoch_1,
            "loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "accuracy": acc,
            "lr": round(lr, 8) if lr is not None else None,
        }
        self.history.append(entry)

        progress = self._phase_progress(epoch_1)
        store.update_task(
            self.task_id,
            progress=progress,
            status="TRAINING",
            current_phase="final_training",
            history=list(self.history),
        )


class TrialProgressCallback:
    """Optuna study callback — maps trial completion to the HPO phase window."""

    def __init__(self, task_id: str, n_trials: int,
                 phase_range: tuple[int, int] = (15, 50)) -> None:
        self.task_id = task_id
        self.n_trials = max(int(n_trials), 1)
        self.phase_start, self.phase_end = phase_range
        self._last = self.phase_start

    def __call__(self, study, trial):
        completed = trial.number + 1
        frac = min(completed / self.n_trials, 1.0)
        raw = self.phase_start + int(round((self.phase_end - self.phase_start) * frac))
        raw = max(raw, self._last)
        raw = min(raw, self.phase_end)
        self._last = raw
        store.update_task(
            self.task_id,
            current_trial=completed,
            total_trials=self.n_trials,
            progress=raw,
            current_phase="hpo",
        )
