"""Unit tests for the SQLite-backed graph sample cache."""
from __future__ import annotations

import json
import pytest

from app.data import graph_cache_sqlite as cache


@pytest.fixture(autouse=True)
def reset_cache():
    cache.reset_for_test()
    yield
    cache.reset_for_test()


def test_sqlite_cache_put_get_roundtrip():
    data = json.dumps({"nodes": [1, 2, 3]}).encode()
    h = cache.content_hash(data)
    cache.put("ds1", "g1", h, data)
    assert cache.get("ds1", "g1", h) == data


def test_sqlite_cache_content_hash_mismatch_returns_none():
    data = json.dumps({"x": 1}).encode()
    cache.put("ds1", "g1", "OLDHASH", data)
    # New hash → mismatch → returns None and deletes entry
    assert cache.get("ds1", "g1", "NEWHASH") is None
    # Entry invalidated; old hash should also return None now
    assert cache.get("ds1", "g1", "OLDHASH") is None


def test_sqlite_cache_invalidate_dataset():
    cache.put("ds1", "g1", "h", b"a")
    cache.put("ds1", "g2", "h", b"b")
    cache.put("ds2", "g1", "h", b"c")
    deleted = cache.invalidate("ds1")
    assert deleted == 2
    assert cache.get("ds1", "g1", "h") is None
    assert cache.get("ds2", "g1", "h") == b"c"


def test_size_bytes_tracks_total():
    cache.put("ds", "g", "h", b"hello")
    assert cache.size_bytes() == 5


def test_get_returns_none_for_missing_entry():
    assert cache.get("nonexistent_ds", "nonexistent_g", "anyhash") is None


def test_put_overwrites_existing_entry():
    data_v1 = b"version1"
    data_v2 = b"version2"
    cache.put("ds1", "g1", "h1", data_v1)
    cache.put("ds1", "g1", "h2", data_v2)
    # Should return v2 with matching hash
    assert cache.get("ds1", "g1", "h2") == data_v2
    # Old hash no longer valid
    assert cache.get("ds1", "g1", "h1") is None
