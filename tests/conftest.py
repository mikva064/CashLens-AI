"""Choose isolated service settings before any test module imports app.config."""
import os
import gc
import tempfile


_service_data = tempfile.TemporaryDirectory(prefix="cashlens-tests-")
os.environ["CASHLENS_DATA_DIR"] = _service_data.name
os.environ["CASHLENS_API_KEY"] = "test-key"
os.environ["CASHLENS_ANALYZER"] = "mock"
os.environ["CASHLENS_MOCK_DELAY"] = "0"


def pytest_unconfigure(config):
    gc.collect()
    _service_data.cleanup()
