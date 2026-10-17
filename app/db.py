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
        # Отметки проверяющего: подтвердил событие или счёл ложным срабатыванием.
        # Это размеченные примеры для дообучения и подбора порогов.
        c.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                job_id TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                decision TEXT NOT NULL,
                comment TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY (job_id, event_index)
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


def set_feedback(job_id: str, event_index: int, decision: str, comment: str | None) -> None:
    with _lock, _conn() as c:
        c.execute(
            """INSERT INTO feedback (job_id, event_index, decision, comment, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (job_id, event_index) DO UPDATE SET
                 decision=excluded.decision, comment=excluded.comment, created_at=excluded.created_at""",
            (job_id, event_index, decision, comment, time.time()),
        )


def get_feedback(job_id: str) -> dict[int, str]:
    with _conn() as c:
        rows = c.execute(
            "SELECT event_index, decision FROM feedback WHERE job_id=?", (job_id,)
        ).fetchall()
    return {r["event_index"]: r["decision"] for r in rows}
