"""Хранилище задач в SQLite: переживает перезапуск сервиса."""

import json
from contextlib import closing
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
    with _lock, closing(_conn()) as c, c:
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
        columns = {row[1] for row in c.execute("PRAGMA table_info(jobs)")}
        for column in ("failure_code", "input_sha256", "result_sha256"):
            if column not in columns:
                c.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
        # Задачи, прерванные перезапуском, возвращаем в очередь.
        c.execute("UPDATE jobs SET status='queued', started_at=NULL WHERE status='processing'")


def create(job_id: str, filename: str, *, created_at: float | None = None,
           max_outstanding: int | None = None, timeout_sec: float | None = None,
           input_sha256: str | None = None) -> bool:
    """Atomically admit a job or persist an immediately readable failure."""
    now = time.time()
    uploaded_at = now if created_at is None else created_at
    with _lock, closing(_conn()) as c, c:
        c.execute("BEGIN IMMEDIATE")
        outstanding = c.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')"
        ).fetchone()[0]
        code = None
        error = None
        if timeout_sec is not None and now - uploaded_at >= timeout_sec:
            code, error = "upload_deadline", "Загрузка заняла весь лимит времени. Повторите отправку файла."
        elif max_outstanding is not None and outstanding >= max_outstanding:
            code, error = "queue_full", "Очередь заполнена. Дождитесь завершения текущего анализа и повторите загрузку."
        c.execute(
            """INSERT INTO jobs (id, status, filename, created_at, finished_at, error, failure_code, input_sha256)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, "failed" if code else "queued", filename, uploaded_at,
             now if code else None, error, code, input_sha256),
        )
        return code is None


def get(job_id: str) -> Optional[dict]:
    with closing(_conn()) as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    job["result"] = json.loads(job["result"]) if job["result"] else None
    return job


def claim_next() -> Optional[str]:
    """Берёт самую старую задачу из очереди и помечает её processing."""
    with _lock, closing(_conn()) as c, c:
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


def finish(job_id: str, result: dict, *, deadline_at: float | None = None,
           result_sha256: str | None = None) -> bool:
    with _lock, closing(_conn()) as c, c:
        now = time.time()
        cursor = c.execute(
            """UPDATE jobs SET status='done', finished_at=?, result=?, result_sha256=?, error=NULL, failure_code=NULL
               WHERE id=? AND status='processing' AND (? IS NULL OR ? > ?)""",
            (now, json.dumps(result, ensure_ascii=False), result_sha256, job_id, deadline_at, deadline_at, now),
        )
        return cursor.rowcount == 1


def fail(job_id: str, error: str, *, code: str | None = None) -> None:
    with _lock, closing(_conn()) as c, c:
        c.execute(
            """UPDATE jobs SET status='failed', finished_at=?, error=?, failure_code=?
               WHERE id=? AND status IN ('queued', 'processing')""",
            (time.time(), error, code, job_id),
        )


def expire_queued(timeout_sec: float) -> int:
    now = time.time()
    with _lock, closing(_conn()) as c, c:
        cursor = c.execute(
            """UPDATE jobs SET status='failed', finished_at=?, error=?, failure_code='queue_deadline'
               WHERE status='queued' AND created_at <= ?""",
            (now, "Превышено время ожидания в очереди", now - timeout_sec),
        )
        return cursor.rowcount


def fail_outstanding(error: str, *, code: str) -> int:
    with _lock, closing(_conn()) as c, c:
        cursor = c.execute(
            """UPDATE jobs SET status='failed', finished_at=?, error=?, failure_code=?
               WHERE status IN ('queued', 'processing')""",
            (time.time(), error, code),
        )
        return cursor.rowcount


def queue_position(job_id: str) -> int:
    with closing(_conn()) as c:
        row = c.execute(
            """SELECT COUNT(*) AS n FROM jobs WHERE status='queued'
               AND created_at < (SELECT created_at FROM jobs WHERE id=?)""",
            (job_id,),
        ).fetchone()
    return row["n"]


def set_feedback(job_id: str, event_index: int, decision: str, comment: str | None) -> None:
    with _lock, closing(_conn()) as c, c:
        c.execute(
            """INSERT INTO feedback (job_id, event_index, decision, comment, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (job_id, event_index) DO UPDATE SET
                 decision=excluded.decision, comment=excluded.comment, created_at=excluded.created_at""",
            (job_id, event_index, decision, comment, time.time()),
        )


def get_feedback(job_id: str) -> dict[int, str]:
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT event_index, decision FROM feedback WHERE job_id=?", (job_id,)
        ).fetchall()
    return {r["event_index"]: r["decision"] for r in rows}
