"""Хранилище задач в SQLite: переживает перезапуск сервиса."""

import json
import sqlite3
import threading
import time
from typing import Optional

from app.config import DB_PATH

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _lock, _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                filename TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                result TEXT,
                error TEXT
            )"""
        )
        # Задачи, прерванные перезапуском, возвращаем в очередь.
        c.execute("UPDATE jobs SET status='queued', started_at=NULL WHERE status='processing'")


def create(job_id: str, filename: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, status, filename, created_at) VALUES (?, 'queued', ?, ?)",
            (job_id, filename, time.time()),
        )


def get(job_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    job["result"] = json.loads(job["result"]) if job["result"] else None
    return job


def claim_next() -> Optional[str]:
    """Берёт самую старую задачу из очереди и помечает её processing."""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        c.execute(
            "UPDATE jobs SET status='processing', started_at=? WHERE id=?",
            (time.time(), row["id"]),
        )
        return row["id"]


def finish(job_id: str, result: dict) -> None:
    with _lock, _conn() as c:
        c.execute(
            "UPDATE jobs SET status='done', finished_at=?, result=? WHERE id=?",
            (time.time(), json.dumps(result, ensure_ascii=False), job_id),
        )


def fail(job_id: str, error: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
            (time.time(), error, job_id),
        )


def queue_position(job_id: str) -> int:
    with _conn() as c:
        row = c.execute(
            """SELECT COUNT(*) AS n FROM jobs WHERE status='queued'
               AND created_at < (SELECT created_at FROM jobs WHERE id=?)""",
            (job_id,),
        ).fetchone()
    return row["n"]
