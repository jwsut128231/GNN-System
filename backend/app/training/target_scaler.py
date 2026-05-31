"""Regression target standardization (scalar OR per-target vector).

Hetero demo labels (e.g. `total_wirelength` ~600) explode MSE and produce
uninterpretable R². Fit a StandardScaler on train-split `y`, apply to train /
val / test during fit; un-scale predictions for metric reporting so metrics
stay in the original target space.

Multi-Y support (2026-05-12): when fit on a 2-D array of shape (N, T) the
scaler stores per-target mean and std as numpy arrays of length T. The
transform / inverse operations broadcast across the last dim, so they work on
tensors shaped (B, T) (graph-level multi-Y) or (N, T) (node-level multi-Y).
For single-Y the scalar code path is preserved unchanged.

For classification tasks an identity scaler is used — `transform` / `inverse`
are no-ops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


ScalarOrArray = Union[float, np.ndarray]


@dataclass
class TargetScaler:
    """StandardScaler-like helper for a 1-D or 2-D target.

    Fields:
        identity : when True, all ops are no-ops (used for classification).
        mean     : scalar float (single-Y) OR ndarray of shape (T,) (multi-Y).
        std      : matching shape to ``mean``.

    The instance is intentionally polymorphic in ``mean`` / ``std`` so the
    single-Y code paths keep operating on plain Python floats.
    """
    identity: bool = True
    mean: ScalarOrArray = 0.0
    std: ScalarOrArray = 1.0

    @classmethod
    def identity_(cls) -> "TargetScaler":
        return cls(identity=True, mean=0.0, std=1.0)

    @classmethod
    def fit(cls, values: Union[np.ndarray, list, torch.Tensor]) -> "TargetScaler":
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return cls.identity_()

        if arr.ndim == 1:
            # Scalar mean/std — single-Y path.
            mean = float(arr.mean())
            std = float(arr.std())
            if std < 1e-8:
                std = 1.0
            return cls(identity=False, mean=mean, std=std)

        if arr.ndim == 2:
            mean_v = arr.mean(axis=0).astype(np.float64)
            std_v = arr.std(axis=0).astype(np.float64)
            # Guard against degenerate (constant) targets.
            std_v = np.where(std_v < 1e-8, 1.0, std_v)
            return cls(identity=False, mean=mean_v, std=std_v)

        raise ValueError(
            f"TargetScaler.fit expects 1-D or 2-D values, got ndim={arr.ndim}."
        )

    # ── helpers ──

    def _mean_tensor(self, like: torch.Tensor) -> torch.Tensor:
        if isinstance(self.mean, np.ndarray):
            return torch.as_tensor(self.mean, dtype=like.dtype, device=like.device)
        return torch.as_tensor(float(self.mean), dtype=like.dtype, device=like.device)

    def _std_tensor(self, like: torch.Tensor) -> torch.Tensor:
        if isinstance(self.std, np.ndarray):
            return torch.as_tensor(self.std, dtype=like.dtype, device=like.device)
        return torch.as_tensor(float(self.std), dtype=like.dtype, device=like.device)

    # ── tensor ops ──

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return t
        return (t - self._mean_tensor(t)) / self._std_tensor(t)

    def inverse_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return t
        return t * self._std_tensor(t) + self._mean_tensor(t)

    # ── numpy ops (for post-evaluation metrics) ──

    def inverse_np(self, a: np.ndarray) -> np.ndarray:
        if self.identity:
            return a
        if isinstance(self.mean, np.ndarray):
            return a * self.std + self.mean  # broadcasts over last dim
        return a * float(self.std) + float(self.mean)

    # ── serialisation (for checkpoint persistence) ──

    def to_dict(self) -> dict:
        mean = self.mean.tolist() if isinstance(self.mean, np.ndarray) else float(self.mean)
        std = self.std.tolist() if isinstance(self.std, np.ndarray) else float(self.std)
        return {"identity": self.identity, "mean": mean, "std": std}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TargetScaler":
        if not d:
            return cls.identity_()
        identity = bool(d.get("identity", True))
        mean_raw = d.get("mean", 0.0)
        std_raw = d.get("std", 1.0)
        if isinstance(mean_raw, list):
            mean: ScalarOrArray = np.asarray(mean_raw, dtype=np.float64)
        else:
            mean = float(mean_raw)
        if isinstance(std_raw, list):
            std: ScalarOrArray = np.asarray(std_raw, dtype=np.float64)
        else:
            std = float(std_raw)
        return cls(identity=identity, mean=mean, std=std)
