"""Actual spawn/process-tree deadlines using synthetic files and no GPU."""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from app import db, worker
from cashlens_ai.contracts import AnalysisResult
from cashlens_ai.errors import ModelUnavailableError


class SyntheticAnalyzer:
    name = "model"

    def load(self):
        marker = Path(os.environ["CASHLENS_TEST_LOAD_MARKER"])
        with marker.open("a", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\n")
        if os.getenv("CASHLENS_TEST_LOAD_MODE") == "hang":
            time.sleep(60)
        if os.getenv("CASHLENS_TEST_LOAD_MODE") == "fail":
            raise RuntimeError("SYNTHETIC_PRIVATE_LOAD_DETAIL")

    def capabilities(self):
        return {"model_name": "Synthetic spawn test", "rule_count": 0, "rules": [],
                "limitations": "Test only", "performance": {"profile": "test"}}

    def analyze(self, video, output_dir, deadline_sec):
        mode = Path(video).read_text(encoding="utf-8")
        if mode == "crash":
            os._exit(17)
        if mode == "unavailable":
            raise ModelUnavailableError("Synthetic poisoned CUDA state")
        if mode in {"descendant", "descendant_crash"}:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
            (Path(video).parent / "descendant.pid").write_text(str(child.pid), encoding="ascii")
            if mode == "descendant_crash":
                os._exit(18)
        if mode in {"hang", "descendant"}:
            time.sleep(60)
        return AnalysisResult(verdict="no_violation", violations=[])

    def close(self):
        pass


def synthetic_factory():
    return SyntheticAnalyzer()


def wait_for(predicate, *, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for synthetic worker state")


def process_running(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        # Linux can briefly retain a dead descendant as a zombie.
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().split()[2] != "Z"
    except ProcessLookupError:
        return False


@pytest.fixture
def running_supervisor(monkeypatch, tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    marker = tmp_path / "loads.txt"
    monkeypatch.setenv("CASHLENS_TEST_LOAD_MARKER", str(marker))
    monkeypatch.delenv("CASHLENS_TEST_LOAD_MODE", raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(worker, "JOBS_DIR", jobs)
    monkeypatch.setattr(worker, "MODEL_LOAD_TIMEOUT_SEC", 5)
    monkeypatch.setattr(worker, "JOB_TIMEOUT_SEC", 2)
    runners = []

    def factory():
        runner = worker.ModelProcess(factory=synthetic_factory)
        runners.append(runner)
        return runner

    monkeypatch.setattr(worker, "_new_process", factory)
    db.init()
    stops = []

    def start(*, ready=True):
        stop = worker.start()
        stops.append(stop)
        if ready:
            wait_for(lambda: worker.snapshot()["ready"])
        return stop

    def submit(mode="clean", *, age=0):
        job_id = uuid.uuid4().hex
        directory = jobs / job_id
        directory.mkdir()
        (directory / "video.mov").write_text(mode, encoding="utf-8")
        db.create(job_id, "synthetic.mov", created_at=time.time() - age)
        return job_id

    yield start, submit, marker, runners
    for stop in stops:
        worker.shutdown(stop)
    assert not any(runner.process.is_alive() for runner in runners)


def terminal(job_id):
    job = db.get(job_id)
    return job if job["status"] in {"done", "failed"} else None


def test_model_is_preloaded_once_and_reused_between_jobs(running_supervisor):
    start, submit, marker, runners = running_supervisor
    start()
    for _ in range(2):
        assert wait_for(lambda jid=submit(): terminal(jid))["status"] == "done"
    assert len(marker.read_text().splitlines()) == len(runners) == 1
    assert worker.snapshot()["performance"] == {"profile": "test"}


@pytest.mark.parametrize("mode,code", [("hang", "hard_timeout"), ("crash", "worker_crash"),
                                      ("unavailable", "model_unavailable")])
def test_hung_or_crashed_process_is_failed_killed_and_replaced(running_supervisor, mode, code):
    start, submit, marker, runners = running_supervisor
    start()
    old_pid = runners[0].process.pid
    began = time.monotonic()
    job_id = submit(mode)
    job = wait_for(lambda: terminal(job_id))
    assert job["status"] == "failed" and job["failure_code"] == code
    assert job["result"] is None
    assert time.monotonic() - began < 3
    wait_for(lambda: len(runners) >= 2 and worker.snapshot()["ready"])
    assert not process_running(old_pid)
    next_id = submit()
    assert wait_for(lambda: terminal(next_id))["status"] == "done"
    assert len(marker.read_text().splitlines()) == 2


@pytest.mark.parametrize("mode", ["descendant", "descendant_crash"])
def test_timeout_or_crash_kills_descendant_processes_too(running_supervisor, mode):
    start, submit, _, runners = running_supervisor
    start()
    job_id = submit(mode)
    path = worker.JOBS_DIR / job_id / "descendant.pid"
    descendant_pid = int(wait_for(lambda: path.exists() and path.read_text().strip()))
    job = wait_for(lambda: terminal(job_id))
    assert job["status"] == "failed"
    wait_for(lambda: not process_running(descendant_pid))
    wait_for(lambda: len(runners) >= 2 and worker.snapshot()["ready"])


def test_queued_age_consumes_the_hard_deadline(running_supervisor):
    start, submit, _, _ = running_supervisor
    start()
    began = time.monotonic()
    job_id = submit("hang", age=1.5)
    result = wait_for(lambda: terminal(job_id))
    assert result["failure_code"] == "hard_timeout"
    assert time.monotonic() - began < 1


@pytest.mark.parametrize("mode", ["fail", "hang"])
def test_loading_failure_fails_waiting_jobs_without_clean_verdict(running_supervisor, monkeypatch, mode):
    start, submit, _, _ = running_supervisor
    monkeypatch.setenv("CASHLENS_TEST_LOAD_MODE", mode)
    monkeypatch.setattr(worker, "MODEL_LOAD_TIMEOUT_SEC", 0.7 if mode == "hang" else 5)
    job_id = submit()
    start(ready=False)
    result = wait_for(lambda: terminal(job_id))
    assert result["status"] == "failed"
    assert result["failure_code"] == "model_unavailable"
    assert result["result"] is None
    assert "SYNTHETIC_PRIVATE" not in result["error"]
    assert worker.snapshot()["status"] == "failed"


def test_shutdown_marks_active_job_failed_and_releases_process(running_supervisor):
    start, submit, _, runners = running_supervisor
    stop = start()
    job_id = submit("hang")
    wait_for(lambda: db.get(job_id)["status"] == "processing")
    worker.shutdown(stop)
    assert db.get(job_id)["failure_code"] == "service_stopped"
    assert not runners[0].process.is_alive()


def test_done_cannot_replace_a_terminal_failure(running_supervisor):
    _, submit, _, _ = running_supervisor
    job_id = submit()
    assert db.claim_next() == job_id
    db.fail(job_id, "test timeout", code="hard_timeout")
    assert db.finish(job_id, {"analysis": {}}, deadline_at=time.time() + 10) is False
    assert db.get(job_id)["failure_code"] == "hard_timeout"


def test_queue_expires_while_model_is_still_loading(running_supervisor, monkeypatch):
    start, submit, _, _ = running_supervisor
    monkeypatch.setenv("CASHLENS_TEST_LOAD_MODE", "hang")
    monkeypatch.setattr(worker, "JOB_TIMEOUT_SEC", 0.25)
    job_id = submit()
    started = time.monotonic()
    start(ready=False)
    result = wait_for(lambda: terminal(job_id))
    assert result["failure_code"] == "queue_deadline"
    assert time.monotonic() - started < 0.8
    assert worker.snapshot()["status"] == "loading"


def test_late_finalization_also_restarts_the_model_process(running_supervisor, monkeypatch):
    start, submit, _, runners = running_supervisor
    monkeypatch.setattr(db, "finish", lambda *args, **kwargs: False)
    start()
    job_id = submit()
    assert wait_for(lambda: terminal(job_id))["failure_code"] == "hard_timeout"
    wait_for(lambda: len(runners) >= 2 and worker.snapshot()["ready"])
