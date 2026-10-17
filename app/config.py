import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

API_KEY = os.getenv("CASHLENS_API_KEY", "dev-key")
DATA_DIR = Path(os.getenv("CASHLENS_DATA_DIR", ROOT / "data"))
RULES_PATH = Path(os.getenv("CASHLENS_RULES", ROOT / "rules" / "rules.json"))
WEB_DIR = ROOT / "web"

MAX_UPLOAD_MB = int(os.getenv("CASHLENS_MAX_UPLOAD_MB", "300"))
ALLOWED_EXT = {".mov", ".mp4"}

# Лимит кейса — 5 минут от загрузки до результата; оставляем запас.
JOB_TIMEOUT_SEC = float(os.getenv("CASHLENS_JOB_TIMEOUT", "270"))

# Веб-форма работает без X-API-Key (ключ не должен попадать в браузер).
WEB_UPLOAD_ENABLED = os.getenv("CASHLENS_WEB_UPLOAD", "1") == "1"

# Файлы задач раздаются через /media, поэтому БД лежит отдельно от них.
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "jobs.sqlite3"
