"""Real-model integration boundaries without weights, GPU, or customer video."""

import contextlib
import json
import logging
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from analyzer import get_analyzer
from analyzer.qwen import QwenAnalyzer
from app import db, main, worker
from cashlens_ai.contracts import AnalysisResult, Rule, RuleSource
from cashlens_ai.errors import (AnalysisTimeoutError, InvalidModelOutputError,
                                ModelUnavailableError, RuleValidationError)
from cashlens_ai.pipeline import AnalyzerConfig, parse_model_response


RULE = Rule(
    id="test_rule", title="Test observable event",
    observable_condition="An observable condition for a synthetic unit test.",
    source=RuleSource(document="Test source", section="1", url="https://example.invalid/rule"),
    verified=True,
)


class StubModel:
    name = "model"

    def __init__(self, *, load_error=None, analyze_error=None, load_gate=None, delay=0):
        self.load_error = load_error
        self.analyze_error = analyze_error
        self.load_gate = load_gate
        self.delay = delay
        self.closed = False

    def load(self):
        if self.load_gate is not None:
            assert self.load_gate.wait(timeout=5)
        if self.load_error:
            raise self.load_error

    def capabilities(self):
        return {"model_name": "Synthetic model stub", "rule_count": 1,
                "rules": [RULE.model_dump(mode="json", exclude={"verified"})],
                "limitations": "Test capability information",
                "performance": {"profile": "balanced", "gpu_duty_cycle": 0.75,
                                "gpu_memory_fraction": 0.80}}

    def analyze(self, video, output_dir, deadline_sec):
        assert Path(video).exists()
        assert deadline_sec > 0
        time.sleep(self.delay)
        if self.analyze_error:
            raise self.analyze_error
        return AnalysisResult(verdict="no_violation", violations=[])

    def close(self):
        self.closed = True


class InlineProcess:
    """Unit-test transport only; actual kill/restart tests use a spawned process."""
    def __init__(self, model):
        self.model = model
        self.messages = queue.Queue()
        self.closed = False

    def start(self):
        def load():
            try:
                self.model.load()
                self.messages.put({"kind": "ready", "analyzer": self.model.name,
                                   "capabilities": self.model.capabilities()})
            except Exception as exc:
                self.messages.put(worker._error_message(exc, loading=True))
        threading.Thread(target=load, daemon=True).start()

    def send(self, command):
        def analyze():
            try:
                path = worker._execute_job(self.model, command)
                self.messages.put({"kind": "done", "job_id": command["job_id"], "result_path": str(path)})
            except Exception as exc:
                self.messages.put(worker._error_message(exc, command["job_id"]))
        threading.Thread(target=analyze, daemon=True).start()

    def poll(self, timeout=0):
        try:
            return self.messages.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self):
        return not self.closed

    def close(self):
        self.closed = True
        self.model.close()


@pytest.fixture
def service(monkeypatch, tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(main, "JOBS_DIR", jobs)
    monkeypatch.setattr(worker, "JOBS_DIR", jobs)
    monkeypatch.setattr(main.media, "probe_duration", lambda _: 2.0)
    monkeypatch.setattr(main.media, "to_browser_mp4", lambda *args: True)

    @contextlib.contextmanager
    def run(model):
        monkeypatch.setattr(worker, "_new_process", lambda: InlineProcess(model))
        with TestClient(main.app) as client:
            yield client, jobs
        assert model.closed

    return run


def wait_state(client, expected):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        health = client.get("/health").json()
        if health["analyzer_state"] == expected:
            return health
        time.sleep(0.01)
    raise AssertionError(health)


def upload(client):
    # Deliberately synthetic bytes: the media decoder is stubbed at this API boundary.
    return client.post("/web/analyze", files={"file": ("synthetic.mov", b"synthetic fixture")})


def wait_job(client, job_id):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        job = client.get(f"/web/jobs/{job_id}").json()
        if job["status"] != "processing":
            return job
        time.sleep(0.02)
    raise AssertionError(job)


def test_real_local_model_is_default(monkeypatch):
    monkeypatch.delenv("CASHLENS_ANALYZER", raising=False)
    assert isinstance(get_analyzer(), QwenAnalyzer)


def test_loading_model_rejects_upload_without_creating_job(service):
    gate = threading.Event()
    with service(StubModel(load_gate=gate)) as (client, jobs):
        try:
            health = wait_state(client, "loading")
            assert health["analyzer_ready"] is False
            response = upload(client)
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "5"
            assert list(jobs.iterdir()) == []
        finally:
            gate.set()
        wait_state(client, "ready")


def test_failed_model_load_is_safe_and_never_accepts_job(service):
    private = "C:/private/model/secret.txt"
    with service(StubModel(load_error=RuntimeError(private))) as (client, jobs):
        health = wait_state(client, "failed")
        assert health["analyzer_ready"] is False
        assert private not in health["analyzer_error"]
        response = upload(client)
        assert response.status_code == 503
        assert private not in response.text
        assert list(jobs.iterdir()) == []


def test_ready_model_health_and_successful_job(service):
    with service(StubModel()) as (client, _):
        health = wait_state(client, "ready")
        assert health["analyzer"] == "model"
        assert health["rule_count"] == len(health["rules"]) == 1
        assert health["rules"][0]["source"]["url"] == RULE.source.url
        assert health["model_name"] == "Synthetic model stub"
        assert isinstance(health["limitations"], str)
        assert health["performance"] == {"profile": "balanced", "gpu_duty_cycle": 0.75,
                                         "gpu_memory_fraction": 0.80}
        accepted = upload(client)
        assert accepted.status_code == 200
        result = wait_job(client, accepted.json()["job_id"])
        assert result["status"] == "done"
        assert result["analyzer"] == "model"
        assert result["verdict"] == "no_violation"
        assert result["violations"] == []


def test_model_inference_error_is_failed_without_a_verdict(service):
    with service(StubModel(analyze_error=ModelUnavailableError("private inference stack"))) as (client, _):
        wait_state(client, "ready")
        result = wait_job(client, upload(client).json()["job_id"])
        assert result["status"] == "failed"
        assert "verdict" not in result
        assert "private inference stack" not in result["error"]


@pytest.mark.parametrize("error_kind", ["raw", "model_validation"])
def test_job_failure_never_logs_scene_descriptions(service, caplog, error_kind):
    scene = "SYNTHETIC_SCENE_47"
    if error_kind == "raw":
        error = RuntimeError(scene)
    else:
        # A structured description makes Pydantic include scene text in the
        # chained ValidationError. Generic worker logs must omit it.
        raw = json.dumps({"events": [{"rule_id": "test_rule", "timecode_sec": 0,
                                      "description": {"scene": scene}}]})
        with pytest.raises(InvalidModelOutputError) as caught:
            parse_model_response(raw, allowed_ids={"test_rule"}, first_sec=0, last_sec=2)
        error = caught.value
        assert scene in str(error.__cause__)

    with caplog.at_level(logging.ERROR, logger="cashlens.worker"):
        with service(StubModel(analyze_error=error)) as (client, _):
            wait_state(client, "ready")
            job_id = upload(client).json()["job_id"]
            result = wait_job(client, job_id)
            stored = db.get(job_id)

    assert result["status"] == stored["status"] == "failed"
    assert "verdict" not in result
    assert result["error"] == stored["error"]
    assert scene not in result["error"]
    assert scene not in caplog.text
    records = [record for record in caplog.records if record.name == "cashlens.worker"
               and record.levelno >= logging.ERROR]
    assert len(records) == 1
    assert type(error).__name__ in records[0].getMessage()
    assert result["error"] in records[0].getMessage()
    assert records[0].exc_info is None


def test_result_after_deadline_is_failed_without_a_verdict(service, monkeypatch):
    monkeypatch.setattr(worker, "JOB_TIMEOUT_SEC", 0.1)
    with service(StubModel(delay=0.2)) as (client, _):
        wait_state(client, "ready")
        result = wait_job(client, upload(client).json()["job_id"])
        assert result["status"] == "failed"
        assert "verdict" not in result


def test_invalid_upload_cleanup(service, monkeypatch):
    monkeypatch.setattr(main.media, "probe_duration", lambda _: None)
    with service(StubModel()) as (client, jobs):
        wait_state(client, "ready")
        assert upload(client).status_code == 422
        assert list(jobs.iterdir()) == []


def test_upload_limit_cleanup(service, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    with service(StubModel()) as (client, jobs):
        wait_state(client, "ready")
        assert upload(client).status_code == 413
        assert list(jobs.iterdir()) == []


def test_upload_cannot_queue_after_permanent_failure_drain(service, monkeypatch):
    with service(StubModel()) as (client, _):
        wait_state(client, "ready")
        monkeypatch.setattr(db, "claim_next", lambda: None)

        def fail_during_probe(_):
            worker._update(ready=False, status="failed", error="Synthetic unavailable model")
            db.fail_outstanding("Synthetic unavailable model", code="model_unavailable")
            return 2.0

        monkeypatch.setattr(main.media, "probe_duration", fail_during_probe)
        response = upload(client)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed" and body["failure_code"] == "model_unavailable"
        assert db.get(body["job_id"])["status"] == "failed"


def test_qwen_uses_root_relative_paths_and_offline_backend(monkeypatch, tmp_path):
    import analyzer.qwen as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setenv("CASHLENS_MODEL_DIR", "model")
    monkeypatch.setenv("CASHLENS_RULES", "config/rules.json")
    monkeypatch.setenv("CASHLENS_SAMPLE_FPS", "0.5")
    monkeypatch.setenv("CASHLENS_MAX_NEW_TOKENS", "128")
    monkeypatch.setattr(module, "load_rules", lambda _: [RULE])
    backend = MagicMock()
    constructor = MagicMock(return_value=backend)
    monkeypatch.setattr(module, "LocalQwenBackend", constructor)
    analyzer = QwenAnalyzer()
    analyzer.load()
    assert analyzer.model_dir == tmp_path / "model"
    assert analyzer.rules_path == tmp_path / "config/rules.json"
    assert analyzer._analyzer.config.sample_fps == 0.5
    assert constructor.call_args.kwargs["max_new_tokens"] == 128
    backend.load.assert_called_once()
    assert analyzer.capabilities()["rule_count"] == 1
    analyzer.close()
    backend.close.assert_called_once()


@pytest.mark.parametrize("profile,expected", [
    (None, ("balanced", 384, 640, 131072, 8)),
    ("quality", ("quality", 448, 896, 262144, 12)),
])
def test_qwen_effective_performance_profiles(monkeypatch, profile, expected):
    import analyzer.qwen as module

    for key in ("CASHLENS_PERFORMANCE_PROFILE", "CASHLENS_MAX_EDGE", "CASHLENS_REFINE_MAX_EDGE",
                "CASHLENS_MAX_PIXELS", "CASHLENS_MAX_REFINE_FRAMES", "CASHLENS_GPU_DUTY_CYCLE",
                "CASHLENS_GPU_MEMORY_FRACTION", "CASHLENS_SAMPLE_FPS", "CASHLENS_WINDOW_SEC",
                "CASHLENS_OVERLAP_SEC"):
        monkeypatch.delenv(key, raising=False)
    if profile is not None:
        monkeypatch.setenv("CASHLENS_PERFORMANCE_PROFILE", profile)
    backend = MagicMock()
    constructor = MagicMock(return_value=backend)
    monkeypatch.setattr(module, "LocalQwenBackend", constructor)
    monkeypatch.setattr(module, "load_rules", lambda _: [RULE])
    analyzer = QwenAnalyzer()
    analyzer.load()
    info = analyzer.capabilities()["performance"]
    assert tuple(info[key] for key in ("profile", "max_edge", "refine_max_edge", "max_pixels",
                                       "max_refine_frames")) == expected
    assert info["sample_fps"] == 1.0
    assert (info["window_sec"], info["overlap_sec"]) == (24.0, 4.0)
    assert info["gpu_duty_cycle"] == constructor.call_args.kwargs["gpu_duty_cycle"] == 0.75
    assert info["gpu_memory_fraction"] == constructor.call_args.kwargs["gpu_memory_fraction"] == 0.80
    assert info["max_pixels"] == constructor.call_args.kwargs["max_pixels"]


def test_qwen_explicit_settings_override_performance_profile(monkeypatch):
    import analyzer.qwen as module

    for key, value in {"CASHLENS_PERFORMANCE_PROFILE": "quality", "CASHLENS_MAX_EDGE": "512",
                       "CASHLENS_REFINE_MAX_EDGE": "768", "CASHLENS_MAX_PIXELS": "196608",
                       "CASHLENS_MAX_REFINE_FRAMES": "6", "CASHLENS_GPU_DUTY_CYCLE": "0.6",
                       "CASHLENS_GPU_MEMORY_FRACTION": "0.7", "CASHLENS_SAMPLE_FPS": "0.5"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(module, "LocalQwenBackend", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(module, "load_rules", lambda _: [RULE])
    analyzer = QwenAnalyzer()
    analyzer.load()
    info = analyzer.capabilities()["performance"]
    assert (info["max_edge"], info["refine_max_edge"], info["max_pixels"],
            info["max_refine_frames"], info["sample_fps"]) == (512, 768, 196608, 6, 0.5)
    assert (info["gpu_duty_cycle"], info["gpu_memory_fraction"]) == (0.6, 0.7)


def test_unknown_performance_profile_blocks_weight_loading(monkeypatch):
    import analyzer.qwen as module

    monkeypatch.setenv("CASHLENS_PERFORMANCE_PROFILE", "unknown")
    constructor = MagicMock()
    monkeypatch.setattr(module, "LocalQwenBackend", constructor)
    with pytest.raises(ValueError, match="balanced or quality"):
        QwenAnalyzer().load()
    constructor.assert_not_called()


@pytest.mark.parametrize("failure", [None, "inference", "budget"])
def test_adapter_releases_unused_memory_and_preserves_loaded_backend(failure):
    adapter = QwenAnalyzer()
    backend = MagicMock()
    adapter._backend = backend
    config = AnalyzerConfig()
    result = AnalysisResult(verdict="no_violation", violations=[])
    analyze = MagicMock(return_value=result)
    adapter._analyzer = SimpleNamespace(config=config, analyze=analyze)
    if failure == "inference":
        analyze.side_effect = AnalysisTimeoutError("synthetic timeout")
    if failure is None:
        assert adapter.analyze(Path("synthetic.mov"), Path("out"), 80) is result
    else:
        with pytest.raises(AnalysisTimeoutError):
            adapter.analyze(Path("synthetic.mov"), Path("out"), 10 if failure == "budget" else 80)
    assert adapter._analyzer.config is config
    assert adapter._backend is backend
    backend.release_unused_memory.assert_called_once_with()
    backend.close.assert_not_called()
    if failure == "budget":
        analyze.assert_not_called()


def test_invalid_rules_block_loading_weights(monkeypatch):
    import analyzer.qwen as module

    backend = MagicMock()
    monkeypatch.setattr(module, "LocalQwenBackend", backend)
    monkeypatch.setattr(module, "load_rules", MagicMock(side_effect=RuleValidationError("invalid")))
    with pytest.raises(RuleValidationError):
        QwenAnalyzer().load()
    backend.assert_not_called()


def test_failed_qwen_load_releases_backend(monkeypatch):
    import analyzer.qwen as module

    backend = MagicMock()
    backend.load.side_effect = ModelUnavailableError("synthetic load failure")
    monkeypatch.setattr(module, "LocalQwenBackend", MagicMock(return_value=backend))
    monkeypatch.setattr(module, "load_rules", lambda _: [RULE])
    with pytest.raises(ModelUnavailableError):
        QwenAnalyzer().load()
    backend.close.assert_called_once()


def test_adapter_rejects_insufficient_budget_and_restores_config():
    adapter = QwenAnalyzer()
    config = AnalyzerConfig()
    analyze = MagicMock(side_effect=AnalysisTimeoutError("synthetic timeout"))
    adapter._analyzer = SimpleNamespace(config=config, analyze=analyze)
    with pytest.raises(AnalysisTimeoutError):
        adapter.analyze(Path("synthetic.mov"), Path("out"), 10)
    analyze.assert_not_called()
    with pytest.raises(AnalysisTimeoutError):
        adapter.analyze(Path("synthetic.mov"), Path("out"), 80)
    assert adapter._analyzer.config is config
