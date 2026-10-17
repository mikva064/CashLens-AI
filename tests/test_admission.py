"""Atomic admission under simultaneous uploads, without starting a model."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

def test_only_one_of_five_requests_is_admitted(tmp_path, monkeypatch):
    from app import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.sqlite3")
    db.init()
    barrier = threading.Barrier(5)

    def submit(index):
        barrier.wait()
        return db.create(str(index), "synthetic.mov", max_outstanding=1)

    with ThreadPoolExecutor(max_workers=5) as pool:
        admitted = list(pool.map(submit, range(5)))
    assert sum(admitted) == 1
    for index, accepted in enumerate(admitted):
        job = db.get(str(index))
        assert job["status"] == ("queued" if accepted else "failed")
        if not accepted:
            assert job["failure_code"] == "queue_full"
            assert job["finished_at"] is not None and job["result"] is None


def test_upload_time_counts_toward_deadline(tmp_path, monkeypatch):
    from app import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.sqlite3")
    db.init()
    assert not db.create("expired", "synthetic.mov", created_at=time.time() - 301,
                         timeout_sec=270, max_outstanding=1)
    job = db.get("expired")
    assert job["status"] == "failed" and job["failure_code"] == "upload_deadline"
    assert job["result"] is None
