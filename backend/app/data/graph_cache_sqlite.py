"""SQLite-backed graph sample cache (JSON BLOB, content-hash invalidation)."""
from __future__ import annotations

import hashlib
import pathlib
import sqlite3
import time
from typing import Optional

_DB_PATH: Optional[pathlib.Path] = None
_CONN: Optional[sqlite3.Connection] = None


def _get_db_path() -> pathlib.Path:
    # Use STORAGE_DIR from settings if available; fallback to data/
    try:
        from app.core.config import settings
        data_dir = getattr(settings, "DATA_DIR", None) or getattr(settings, "STORAGE_DIR", None)
    except Exception:
        data_dir = None

    if data_dir is None:
        data_dir = pathlib.Path("data")
    if isinstance(data_dir, str):
        data_dir = pathlib.Path(data_dir)

    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "graph_cache.db"


def _get_conn() -> sqlite3.Connection:
    global _CONN, _DB_PATH
    if _CONN is None:
        _DB_PATH = _get_db_path()
        _CONN = sqlite3.connect(str(_DB_PATH), timeout=5.0, check_same_thread=False)
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.execute("""
            CREATE TABLE IF NOT EXISTS graph_cache (
                dataset_id TEXT NOT NULL,
                graph_id   TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                data       BLOB NOT NULL,
                cached_at  INTEGER NOT NULL,
                PRIMARY KEY (dataset_id, graph_id)
            )
        """)
        _CONN.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset ON graph_cache(dataset_id)"
        )
        _CONN.commit()
    return _CONN


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def get(dataset_id: str, graph_id: str, expected_hash: str) -> Optional[bytes]:
    """Return cached JSON bytes if hash matches; None otherwise.

    If the stored hash mismatches, the stale entry is deleted immediately.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT content_hash, data FROM graph_cache"
        " WHERE dataset_id = ? AND graph_id = ?",
        (dataset_id, graph_id),
    ).fetchone()
    if row is None:
        return None
    cached_hash, cached_data = row
    if cached_hash != expected_hash:
        conn.execute(
            "DELETE FROM graph_cache WHERE dataset_id = ? AND graph_id = ?",
            (dataset_id, graph_id),
        )
        conn.commit()
        return None
    return bytes(cached_data)


def put(dataset_id: str, graph_id: str, content_hash_value: str, data: bytes) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO graph_cache"
        " (dataset_id, graph_id, content_hash, data, cached_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (dataset_id, graph_id, content_hash_value, data, int(time.time())),
    )
    conn.commit()


def invalidate(dataset_id: str) -> int:
    """Remove all cache entries for a dataset. Return the number of rows deleted."""
    conn = _get_conn()
    cur = conn.execute(
        "DELETE FROM graph_cache WHERE dataset_id = ?", (dataset_id,)
    )
    conn.commit()
    return cur.rowcount


def size_bytes() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT SUM(LENGTH(data)) FROM graph_cache").fetchone()
    return row[0] or 0


def reset_for_test() -> None:
    """Test helper: close connection and delete the DB file (and WAL/SHM)."""
    global _CONN, _DB_PATH
    if _CONN is not None:
        _CONN.close()
        _CONN = None
    if _DB_PATH and _DB_PATH.exists():
        _DB_PATH.unlink()
        wal = _DB_PATH.with_suffix(".db-wal")
        shm = _DB_PATH.with_suffix(".db-shm")
        if wal.exists():
            wal.unlink()
        if shm.exists():
            shm.unlink()
    _DB_PATH = None
