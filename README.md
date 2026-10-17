# CashLens AI

Сервис для выявления признаков нарушений кассовой дисциплины по видео.
Загрузка `.mov` → поиск событий → вердикт, таймкоды, кадры-доказательства и
нормативный источник РК для каждого события.

> Статус: каркас сервиса работает на **заглушке анализатора**
> (`CASHLENS_ANALYZER=mock`). Модель подключается через `analyzer/model.py`,
> см. [analyzer/contract.md](analyzer/contract.md).

## Запуск локально

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

- Интерфейс: http://localhost:8000/
- Документация API: http://localhost:8000/docs
- Тесты: `.venv/Scripts/python -m pytest -q`

Отдельно ставить ffmpeg не нужно: если его нет в PATH, берётся бинарник из
пакета `imageio-ffmpeg`.

Настройки — переменные окружения, см. [.env.example](.env.example).

## API

Все запросы `/api/v1/*` требуют заголовок `X-API-Key`.

| Метод | Путь | Ответ |
|---|---|---|
| `POST` | `/api/v1/analyze` | `multipart/form-data`, поле `file` → `202 {"job_id", "status": "queued"}` |
| `GET` | `/api/v1/jobs/{job_id}` | статус и, когда `done`, результат |
| `GET` | `/health` | проверка живости |

```bash
curl -H "X-API-Key: dev-key" -F "file=@X-4.mov" http://localhost:8000/api/v1/analyze
curl -H "X-API-Key: dev-key" http://localhost:8000/api/v1/jobs/<job_id>
```

Статусы задачи: `queued` → `processing` → `done` | `failed`.

Результат (`done`):

```json
{
  "job_id": "…",
  "status": "done",
  "verdict": "possible_violation",
  "processing_time_sec": 41.3,
  "duration_sec": 57.0,
  "video_url": "/media/<job_id>/preview.mp4",
  "events": [
    {
      "type": "…",
      "type_title": "…",
      "start_sec": 55.0,
      "end_sec": 59.5,
      "timecode": "00:57",
      "confidence": 0.78,
      "description": "…",
      "evidence_frames": [{"sec": 55.0, "url": "/media/<job_id>/e1_1.jpg"}],
      "rule": {"id": "…", "title": "…", "requirement": "…", "source": "…", "url": "…", "verified": true}
    }
  ]
}
```

`verdict`: `no_violation` — нарушение не обнаружено, `possible_violation` —
обнаружены признаки возможного нарушения.

## Архитектура

```
POST → проверка файла (расширение, размер, ffmpeg) → data/jobs/<id>/video.mov
     → SQLite: queued → воркер (один, по очереди) → analyzer.analyze_video()
     → кадры «до / во время / после» (ffmpeg) + норма из rules/rules.json
     → preview.mp4 для браузера → SQLite: done
```

- `app/` — FastAPI, очередь, воркер, ffmpeg.
- `analyzer/` — анализатор видео; сервис зависит только от контракта.
- `rules/rules.json` — справочник «тип события → требование → источник».
  Текст нормы берётся отсюда, а не генерируется моделью.
- `web/` — одностраничный интерфейс (без фреймворков).

Лимит 5 минут считается от момента загрузки, включая ожидание в очереди
(`CASHLENS_JOB_TIMEOUT`, по умолчанию 270 с). Анализатор получает
оставшееся время как `deadline_sec`.

## Ограничения

- Видео и кадры не передаются во внешние сервисы; идентификация лиц не выполняется.
- Справочник правил пока содержит только тестовый тип.
