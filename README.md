# CashLens AI

Сервис для выявления признаков нарушений кассовой дисциплины по видео.
Загрузка `.mov` → поиск событий → вердикт, таймкоды, кадры-доказательства и
нормативный источник РК для каждого события.

> Статус: каркас сервиса работает на **заглушке анализатора**
> (`CASHLENS_ANALYZER=mock`). Модель подключается через `analyzer/model.py`,
> см. [analyzer/contract.md](analyzer/contract.md).

## Запуск локально

Windows: двойной клик по `run.bat` (сам создаст окружение при первом запуске),
затем открыть http://localhost:8000.

Вручную:

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

Схема — по ТЗ, раздел 10. Все запросы `/api/v1/*` требуют заголовок `X-API-Key`.

| Метод | Путь | Ответ |
|---|---|---|
| `POST` | `/api/v1/analyze` | `multipart/form-data`, поле `file` (.mov) → `{"job_id", "status": "processing"}` |
| `GET` | `/api/v1/jobs/{job_id}` | статус `processing` / `done` / `failed` и результат |
| `GET` | `/health` | проверка живости |

```bash
curl -X POST -H "X-API-Key: dev-key" -F "file=@video.mov" http://localhost:8000/api/v1/analyze
curl -H "X-API-Key: dev-key" http://localhost:8000/api/v1/jobs/<job_id>
```

Результат (`done`):

```json
{
  "job_id": "3f2a9c1e…",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Название типа нарушения",
      "timecode_sec": 44.2,
      "description": "Что видно на видео и почему это может быть нарушением. Требование: … (источник).",
      "evidence_frames": [
        {"timestamp_sec": 44.2, "image_base64": "/9j/4AAQ…"}
      ],
      "rule": {"id": "…", "title": "…", "requirement": "…", "source": "…", "url": "…", "verified": true},
      "start_sec": 42.0,
      "end_sec": 46.5,
      "confidence": 0.78
    }
  ],
  "processing_time_sec": 41.3
}
```

- `verdict`: `no_violation` или `violation`; `violations` пустой, если нарушений нет.
- `evidence_frames`: 1–3 JPEG в base64; первый кадр — ровно на `timecode_sec`.
- Поля `rule`, `start_sec`, `end_sec`, `confidence`, `stage`, `timings` —
  дополнительные, схему ТЗ не нарушают.
- Задача в очереди отдаётся как `status: "processing"` с `stage: "queued"`.

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
