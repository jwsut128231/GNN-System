"""Unit tests for TargetScaler — pure numpy/torch, no Lightning required."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.training.target_scaler import TargetScaler  # noqa: E402


def test_identity_scaler_is_noop():
    s = TargetScaler.identity_()
    t = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(s.transform_tensor(t), t)
    assert torch.equal(s.inverse_tensor(t), t)
    assert np.array_equal(s.inverse_np(np.array([1.0, 2.0])), np.array([1.0, 2.0]))


def test_fit_and_roundtrip():
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    s = TargetScaler.fit(y)
    assert not s.identity
    scaled = s.transform_tensor(torch.tensor(y, dtype=torch.float))
    # mean ≈ 0, std ≈ 1
    assert abs(float(scaled.mean().item())) < 1e-5
    assert abs(float(scaled.std(unbiased=False).item()) - 1.0) < 1e-5
    # inverse restores
    restored = s.inverse_tensor(scaled)
    assert torch.allclose(restored, torch.tensor(y, dtype=torch.float), atol=1e-4)


def test_constant_target_does_not_divide_by_zero():
    s = TargetScaler.fit(np.array([3.0, 3.0, 3.0]))
    # std is clamped to 1.0 when near-zero — no NaN/Inf
    t = s.transform_tensor(torch.tensor([3.0, 3.0]))
    assert torch.isfinite(t).all()


def test_dict_roundtrip():
    s = TargetScaler.fit(np.arange(10).astype(float))
    d = s.to_dict()
    s2 = TargetScaler.from_dict(d)
    assert s2.identity == s.identity
    assert abs(s2.mean - s.mean) < 1e-9
    assert abs(s2.std - s.std) < 1e-9


def test_from_dict_none_returns_identity():
    s = TargetScaler.from_dict(None)
    assert s.identity


# ── Multi-Y vector targets (2026-05-12) ────────────────────────────────

def test_fit_vector_targets_stores_per_target_mean_std():
    # Two targets — first ~100, second ~10.
    y = np.array([
        [100.0, 10.0],
        [200.0, 20.0],
        [300.0, 30.0],
        [400.0, 40.0],
    ])
    s = TargetScaler.fit(y)
    assert not s.identity
    assert isinstance(s.mean, np.ndarray)
    assert s.mean.shape == (2,)
    np.testing.assert_allclose(s.mean, [250.0, 25.0])
    np.testing.assert_allclose(s.std, [np.std([100, 200, 300, 400]),
                                       np.std([10, 20, 30, 40])])


def test_transform_vector_targets_broadcasts():
    y = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    s = TargetScaler.fit(y)
    t = torch.tensor(y, dtype=torch.float)
    scaled = s.transform_tensor(t)
    # per-column mean ≈ 0 after scaling
    for col in range(2):
        assert abs(float(scaled[:, col].mean().item())) < 1e-5
    # round-trip
    restored = s.inverse_tensor(scaled)
    assert torch.allclose(restored, t, atol=1e-3)


def test_inverse_np_vector_broadcasts():
    y = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    s = TargetScaler.fit(y)
    a_scaled = (y - s.mean) / s.std
    a_unscaled = s.inverse_np(a_scaled)
    np.testing.assert_allclose(a_unscaled, y, atol=1e-6)


def test_dict_roundtrip_vector():
    y = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    s = TargetScaler.fit(y)
    d = s.to_dict()
    assert isinstance(d["mean"], list) and isinstance(d["std"], list)
    assert len(d["mean"]) == 2
    s2 = TargetScaler.from_dict(d)
    assert isinstance(s2.mean, np.ndarray)
    np.testing.assert_allclose(s2.mean, s.mean)
    np.testing.assert_allclose(s2.std, s.std)


def test_fit_constant_vector_target_clamps_std():
    y = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])  # col 0 constant
    s = TargetScaler.fit(y)
    # First column std should be clamped to 1.0 (no division by zero).
    assert s.std[0] == 1.0
    # Second column should still scale properly.
    assert s.std[1] > 0


def test_3d_input_rejected():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        TargetScaler.fit(np.zeros((2, 2, 2)))
