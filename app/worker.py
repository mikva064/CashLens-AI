"""Parent-owned jobs and a persistent, killable, preloaded model process."""
import json
import logging
import multiprocessing
import os
import threading
import time
from pathlib import Path

from analyzer import get_analyzer
from app import db, media, reporting
from app.config import JOBS_DIR, JOB_TIMEOUT_SEC, MODEL_LOAD_TIMEOUT_SEC
from app.process_guard import ProcessTreeGuard, initialize_child_process_group
from cashlens_ai.contracts import AnalysisResult
from cashlens_ai.errors import (AnalysisTimeoutError, InvalidModelOutputError,
                                ModelUnavailableError, RuleValidationError, VideoReadError)

log = logging.getLogger("cashlens.worker")
state = {"analyzer": None, "ready": False, "error": None, "status": "stopped",
         "model_name": None, "rule_count": 0, "rules": [], "limitations": "", "performance": {}}
_state_lock = threading.Lock()
_thread: threading.Thread | None = None
_POLL_SEC = 0.05


def _update(**values) -> None:
    with _state_lock:
        state.update(values)


def snapshot() -> dict:
    with _state_lock:
        return dict(state)


def _public_error(exc: Exception, *, loading: bool = False) -> str:
    if isinstance(exc, RuleValidationError):
        return "Реестр проверенных правил не настроен. Проверьте конфигурацию сервиса."
    if isinstance(exc, AnalysisTimeoutError):
        return "Обработка не уложилась в лимит времени. Полный результат не получен."
    if isinstance(exc, InvalidModelOutputError):
        return "Не удалось проверить ответ модели. Вердикт не сформирован; попробуйте повторить анализ."
    if isinstance(exc, VideoReadError):
        return "Не удалось прочитать видео. Проверьте файл и повторите загрузку."
    if isinstance(exc, ModelUnavailableError):
        return ("Локальная модель недоступна. Проверьте наличие весов, CUDA и свободную видеопамять. "
                "Подробности доступны в журнале сервиса.")
    if loading:
        return "Не удалось запустить локальную модель. Подробности доступны в журнале сервиса."
    return "Ошибка обработки видео. Вердикт не сформирован; подробности доступны в журнале сервиса."


def _error_message(exc: Exception, job_id: str | None = None, *, loading: bool = False) -> dict:
    return {"kind": "load_failed" if loading else "failed", "job_id": job_id,
            "error_type": type(exc).__name__, "error": _public_error(exc, loading=loading)}


def _execute_job(analyzer, command: dict) -> Path:
    """Child only: no SQLite writes or scene text in control messages."""
    job_dir = Path(command["job_dir"])
    src = job_dir / "video.mov"
    if analyzer.name == "mock":
        for key in ("CASHLENS_MOCK_DELAY", "CASHLENS_MOCK_VERDICT"):
            value = command.get("mock_settings", {}).get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    def make_preview():
        try:
            media.to_browser_mp4(src, job_dir / "preview.mp4")
        except Exception as exc:
            log.warning("Preview creation failed (%s)", type(exc).__name__)
    preview = threading.Thread(target=make_preview, daemon=True)
    preview.start()
    try:
        result = analyzer.analyze(src, job_dir / "ai", command["budget_sec"])
        duration = media.probe_duration(src)
        clips = {}
        for index, event in enumerate(result.violations):
            filename = f"event_{index + 1:03d}.mp4"
            bounds = media.make_event_clip(src, event.timecode_sec, job_dir / "clips" / filename,
                                           duration_sec=duration)
            clips[str(index)] = {"filename": filename, **bounds}
        record = {"analysis": result.model_dump(mode="json"), "analyzer": analyzer.name,
                  "duration_sec": duration, "clips": clips}
    finally:
        # No preview is allowed to accumulate behind successful or failed jobs.
        # A stuck conversion is bounded by the parent's hard process deadline.
        preview.join()
    path = job_dir / "worker-result.json"
    temporary = path.with_suffix(".part.json")
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def _child_main(connection, gate, factory=None) -> None:
    initialize_child_process_group()
    if not gate.wait(timeout=15):
        return
    analyzer = None
    try:
        try:
            analyzer = (factory or get_analyzer)()
            analyzer.load()
            connection.send({"kind": "ready", "analyzer": analyzer.name,
                             "capabilities": getattr(analyzer, "capabilities", lambda: {})()})
        except Exception as exc:
            connection.send(_error_message(exc, loading=True))
            return
        while True:
            command = connection.recv()
            if command.get("kind") == "stop":
                return
            job_id = command["job_id"]
            try:
                path = _execute_job(analyzer, command)
                connection.send({"kind": "done", "job_id": job_id, "result_path": str(path)})
            except Exception as exc:
                connection.send(_error_message(exc, job_id))
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        close = getattr(analyzer, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
        connection.close()


class ModelProcess:
    """Spawn keeps Torch and CUDA entirely outside the API parent."""
    def __init__(self, *, factory=None) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, self._child_connection = context.Pipe()
        self._gate = context.Event()
        self.process = context.Process(target=_child_main,
                                       args=(self._child_connection, self._gate, factory),
                                       name="cashlens-model", daemon=True)
        self.guard = ProcessTreeGuard()
        self.closed = False

    def start(self) -> None:
        try:
            self.process.start()
            self._child_connection.close()
            self.guard.attach(self.process.pid)
            self._gate.set()
        except Exception:
            self.close()
            raise

    def poll(self, timeout: float = 0) -> dict | None:
        try:
            return self.connection.recv() if self.connection.poll(timeout) else None
        except (EOFError, BrokenPipeError, OSError):
            return {"kind": "exited"}

    def send(self, message: dict) -> None:
        self.connection.send(message)

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.pid is not None:
                try:
                    self.send({"kind": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(timeout=0.1)
                self.guard.terminate(self.process)
        finally:
            self.guard.close()
            self.connection.close()
            self._child_connection.close()


def _new_process():
    return ModelProcess()


def _expire_queue() -> None:
    db.expire_queued(JOB_TIMEOUT_SEC)


def _wait_ready(stop: threading.Event, runner) -> bool:
    until = time.monotonic() + MODEL_LOAD_TIMEOUT_SEC
    while not stop.is_set():
        _expire_queue()
        if time.monotonic() >= until:
            _update(ready=False, status="failed", error="Превышено время загрузки локальной модели.")
            return False
        message = runner.poll(_POLL_SEC)
        if message and message.get("kind") == "ready":
            allowed = {key: value for key, value in message.get("capabilities", {}).items()
                       if key in {"model_name", "rule_count", "rules", "limitations", "performance"}}
            _update(**allowed, analyzer=message["analyzer"], ready=True, status="ready", error=None)
            log.info("analyzer %s loaded in model process", message["analyzer"])
            return True
        if message and message.get("kind") == "load_failed":
            _update(ready=False, status="failed", error=message["error"])
            log.error("analyzer loading failed (%s)", message["error_type"])
            return False
        if (message and message.get("kind") == "exited") or not runner.is_alive():
            _update(ready=False, status="failed", error="Процесс локальной модели завершился при загрузке.")
            return False
    return False


def _finalize_job(job_id: str, result_path: Path, processing_started_at: float) -> bool:
    """Parent finalizer hook for validated results, web clips, hashes, then DB."""
    job = db.get(job_id)
    job_dir = (JOBS_DIR / job_id).resolve()
    expected = job_dir / "worker-result.json"
    if result_path.resolve() != expected or expected.stat().st_size > 4 * 1024 * 1024:
        raise InvalidModelOutputError("Unexpected model result artifact")
    stored = json.loads(expected.read_text(encoding="utf-8"))
    result = AnalysisResult.model_validate(stored["analysis"])
    for event in result.violations:
        for frame in event.evidence_frames:
            path = frame.image_path.resolve(strict=True)
            if not path.is_relative_to(job_dir / "ai") or not path.is_file():
                raise InvalidModelOutputError("Evidence is outside this job")
    clips = stored.get("clips", {})
    for index in range(len(result.violations)):
        filename = f"event_{index + 1:03d}.mp4"
        if (clips.get(str(index), {}).get("filename") != filename
                or not (job_dir / "clips" / filename).is_file()):
            raise InvalidModelOutputError("Event clip is missing")
    stored["analysis"] = result.model_dump(mode="json")
    stored["timings"] = {"queue_sec": round(processing_started_at - job["created_at"], 2),
                         "processing_sec": round(time.time() - processing_started_at, 2)}
    stored["completed_at"] = time.time()
    report_hash = reporting.write_result_report({**job, "result": stored}, JOBS_DIR)
    if not db.finish(job_id, stored, deadline_at=job["created_at"] + JOB_TIMEOUT_SEC,
                     result_sha256=report_hash):
        (job_dir / "result.json").unlink(missing_ok=True)
        db.fail(job_id, _public_error(AnalysisTimeoutError()), code="hard_timeout")
        return False
    log.info("job %s done: %s, %d violations, %.1fs", job_id,
             result.verdict, len(result.violations), time.time() - processing_started_at)
    return True


def _process_job(stop: threading.Event, runner, job_id: str) -> bool:
    job = db.get(job_id)
    started = time.time()
    budget = JOB_TIMEOUT_SEC - (started - job["created_at"])
    until = time.monotonic() + max(0, budget)
    if budget <= 0:
        db.fail(job_id, "Превышено время ожидания в очереди", code="queue_deadline")
        return True
    runner.send({"kind": "analyze", "job_id": job_id,
                 "job_dir": str((JOBS_DIR / job_id).resolve()), "budget_sec": budget,
                 "mock_settings": {key: os.getenv(key) for key in
                                   ("CASHLENS_MOCK_DELAY", "CASHLENS_MOCK_VERDICT")}})
    while True:
        _expire_queue()
        if stop.is_set():
            db.fail(job_id, "Сервис остановлен; анализ прерван.", code="service_stopped")
            return False
        remaining = until - time.monotonic()
        if remaining <= 0:
            db.fail(job_id, _public_error(AnalysisTimeoutError()), code="hard_timeout")
            _update(ready=False, status="loading", error=None)
            return False
        message = runner.poll(min(_POLL_SEC, remaining))
        if (message and message.get("kind") == "exited") or not runner.is_alive():
            db.fail(job_id, "Процесс модели аварийно завершился; результат не получен.", code="worker_crash")
            _update(ready=False, status="loading", error=None)
            return False
        if not message or message.get("job_id") != job_id:
            continue
        if message["kind"] == "failed":
            log.error("analysis failed (%s): %s", message["error_type"], message["error"])
            restart_codes = {"ModelUnavailableError": "model_unavailable",
                             "AnalysisTimeoutError": "analysis_timeout"}
            code = restart_codes.get(message["error_type"], "analysis_failed")
            db.fail(job_id, message["error"], code=code)
            if message["error_type"] in restart_codes:
                _update(ready=False, status="loading", error=None)
                return False
            return True
        if message["kind"] == "done":
            if time.monotonic() >= until:
                db.fail(job_id, _public_error(AnalysisTimeoutError()), code="hard_timeout")
                _update(ready=False, status="loading", error=None)
                return False
            try:
                return _finalize_job(job_id, Path(message["result_path"]), started)
            except Exception as exc:
                error = _public_error(exc)
                log.error("analysis finalization failed (%s): %s", type(exc).__name__, error)
                db.fail(job_id, error, code="invalid_result")
            return True


def _loop(stop: threading.Event) -> None:
    runner = None
    try:
        while not stop.is_set():
            if runner is None:
                _update(ready=False, status="loading", error=None)
                runner = _new_process()
                runner.start()
                if not _wait_ready(stop, runner):
                    return
            _expire_queue()
            job_id = db.claim_next()
            if job_id is not None:
                try:
                    reusable = _process_job(stop, runner, job_id)
                except (BrokenPipeError, EOFError, OSError):
                    db.fail(job_id, "Связь с процессом модели потеряна.", code="worker_crash")
                    reusable = False
                if not reusable:
                    _update(ready=False, status="loading", error=None)
                    runner.close()
                    runner = None
                continue
            if not runner.is_alive():
                _update(ready=False, status="loading", error=None)
                runner.close()
                runner = None
                continue
            stop.wait(_POLL_SEC)
    except Exception as exc:
        log.error("model supervisor failed (%s)", type(exc).__name__)
        _update(ready=False, status="failed", error=_public_error(exc, loading=True))
    finally:
        if runner is not None:
            runner.close()
        if stop.is_set():
            _update(ready=False, status="stopped")
        elif snapshot()["status"] == "failed":
            db.fail_outstanding(snapshot()["error"] or "Локальная модель недоступна.", code="model_unavailable")


def start() -> threading.Event:
    global _thread
    if _thread is not None and _thread.is_alive():
        raise RuntimeError("The previous analyzer worker is still running")
    _update(analyzer=os.getenv("CASHLENS_ANALYZER", "model").strip().lower(), ready=False,
            error=None, status="loading", model_name=None, rule_count=0, rules=[], limitations="", performance={})
    stop = threading.Event()
    _thread = threading.Thread(target=_loop, args=(stop,), name="cashlens-supervisor", daemon=True)
    _thread.start()
    return stop


def shutdown(stop: threading.Event) -> None:
    stop.set()
    if _thread is not None:
        _thread.join(timeout=3)
