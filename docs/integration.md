# Подключение AI к серверу

Сервер отвечает за загрузку видео, API, X-API-Key, статусы, очередь, очистку и публичную доступность. AI принимает путь к загруженному файлу.

```python
from pathlib import Path
from cashlens_ai.backend import LocalQwenBackend
from cashlens_ai.integration import to_api_payload
from cashlens_ai.pipeline import VideoAnalyzer
from cashlens_ai.rules import load_rules

rules = load_rules("config/rules.json")
backend = LocalQwenBackend(Path("local/models/qwen3-vl-2b"))
backend.load()  # до приёма запросов, без сети
analyzer = VideoAnalyzer(backend, rules)

def run_job(job_id, uploaded_video_path):
    # job_id создаёт сервер; клиент не должен задавать путь output.
    output = Path("local/runs") / job_id
    result = analyzer.analyze(uploaded_video_path, output)
    return to_api_payload(result, job_id=job_id, evidence_root=output)
```

Вызовы одного GPU worker выполняются последовательно. Ошибки `CashLensError`, чтения файлов или валидации сервер преобразует в `status="failed"`. При сбое нельзя выдавать `no_violation`.

Внутренний результат: `schema_version`, `verdict`, `violations`. В событии: `type`, `timecode_sec`, `description`, `rule_id`, проверенный `source` и 1–3 элемента `evidence_frames` с `timestamp_sec`/`image_path`.

`to_api_payload` возвращает требуемый JSON: `job_id`, `status="done"`, `verdict`, `violations`. Кадры превращаются в JPEG base64 (`image_base64`). Источник добавляется в `description`: отдельного поля источника в опубликованной схеме нет. UI может показать структурированный источник из внутреннего результата.

При `no_violation` список событий пуст. Охват определяется активными правилами; автоматически все типы нарушений не покрываются.

Внутренние 240 секунд — кооперативный бюджет. Для пяти минут на весь запрос нужен watchdog отдельного процесса с учётом приёма файла, очереди, сериализации и передачи ответа. Ограничьте очередь на одном GPU. После принудительной остановки worker необходимо восстановить.

Каждому заданию нужен новый output-каталог. Не возвращайте пути файлов наружу и не принимайте их от клиента. Адаптер читает доказательства только внутри каталога конкретного задания.
