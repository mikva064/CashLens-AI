# CashLens AI

Локальный анализ видео кассовых операций: события, исходные таймкоды, кадры-доказательства и заранее проверенные нормативные источники.

Репозиторий состоит из двух частей: AI-модуль `src/cashlens_ai/` (анализ видео) и сервис `app/` + `web/` (веб-интерфейс, HTTP API для жюри, очередь, ключ). Контракт между ними — [docs/integration.md](docs/integration.md).

Реализованы аудит, разделение по уникальным видео, локальный просмотрщик, декодирование с исходными PTS, Qwen backend, двухэтапный анализ и адаптер результата API. **Типы событий ещё нужно определить по локальному просмотру, нормативные связи проверить, качество измерить.** Пустой реестр правил блокирует классификацию и не означает отсутствие нарушений.

## Сервис: веб-интерфейс и API

### Запуск

Windows: двойной клик по `run.bat` (при первом запуске создаст `.venv` на Python 3.12),
затем открыть http://localhost:8000. Документация API — http://localhost:8000/docs.

Режим анализатора задаётся `CASHLENS_ANALYZER`:

- `mock` (по умолчанию) — демо-заглушка без GPU; интерфейс и ответ API помечают
  результат как демо (`"analyzer": "mock"`).
- `model` — локальная Qwen-модель. Нужны веса в `local/models/qwen3-vl-2b`
  (`CASHLENS_MODEL_DIR`) и непустой проверенный `config/rules.json`. Если чего-то
  нет, `/health` возвращает `degraded` с причиной, а задачи получают `failed` —
  ложный `no_violation` не выдаётся.

Остальные настройки — в [.env.example](.env.example). Тесты: `.venv/Scripts/python -m pytest -q tests`.

### API (ТЗ, раздел 10)

Запросы `/api/v1/*` требуют заголовок `X-API-Key` (`CASHLENS_API_KEY`).

| Метод | Путь | Ответ |
|---|---|---|
| `POST` | `/api/v1/analyze` | `multipart/form-data`, поле `file` (.mov) → `{"job_id", "status": "processing"}` |
| `GET` | `/api/v1/jobs/{job_id}` | `processing` / `done` / `failed`; при `done` — результат |
| `GET` | `/health` | готовность анализатора |

```bash
curl -X POST -H "X-API-Key: <ключ>" -F "file=@video.mov" http://localhost:8000/api/v1/analyze
curl -H "X-API-Key: <ключ>" http://localhost:8000/api/v1/jobs/<job_id>
```

Результат `done` собирает `cashlens_ai.integration.to_api_payload` — ровно схема
жюри (`verdict`, `violations[].type / timecode_sec / description / evidence_frames[]`
с JPEG в base64). Дополнительно: `rule_id`, `source` у события, `analyzer`,
`processing_time_sec`, `timings`. Задача в очереди отдаётся как `processing` с
`stage: "queued"`.

### Как устроено

```text
POST → проверка файла → data/jobs/<id>/video.mov → SQLite: queued
     → один воркер (модель загружена один раз) → analyzer.analyze(video, data/jobs/<id>/ai, остаток времени)
     → AnalysisResult (кадры-доказательства пишет AI-модуль) → SQLite: done
     → API: to_api_payload (base64) | веб: ссылки на кадры + mp4-превью
```

- `app/` — FastAPI, очередь на SQLite, воркер, ffmpeg (превью для браузера).
- `analyzer/` — подключение AI-модуля: `qwen.py` (модель) и `mock.py` (демо).
- `web/` — одностраничный интерфейс.

Лимит 5 минут считается от загрузки (`CASHLENS_JOB_TIMEOUT`, 270 с); модель
получает остаток минус 20 с запаса. Ограничение известно: бюджет кооперативный,
жёсткого watchdog-процесса пока нет.

## Установка AI-модуля

Нужен Python 3.12. Подготовка данных работает без GPU; модель требует NVIDIA CUDA. Начальная модель — Qwen3-VL-2B-Instruct в NF4. Совместимость конкретной среды проверяет синтетический smoke test.

```powershell
# Только данные, просмотрщик и тесты
./scripts/setup_windows.ps1 -PythonExe python
# Полная среда с CUDA PyTorch
./scripts/setup_windows.ps1 -PythonExe python -WithModel
.venv/Scripts/python.exe -m cashlens_ai.cli doctor
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Библиотеки и веса скачиваются при подготовке. Анализ использует локальные файлы и не обращается к облачным AI. Команды далее запускаются из корня репозитория.

## Данные и локальная разметка

Замените примеры путей своими. Оригинальные ролики и таблица остаются на диске:

```powershell
.venv/Scripts/python.exe -m cashlens_ai.cli audit --labels 'C:/dataset/open_labels.xlsx' --videos 'C:/dataset/videos' --output local/audit
.venv/Scripts/python.exe -m cashlens_ai.cli review --manifest local/audit/manifest.json --output local/review.html
```

Откройте `local/review.html` непосредственно в браузере. Он открывает оригиналы через file URL, не использует внешние ресурсы и не загружает видео на сервер.

Аудит создаёт `manifest.json` с исходными аннотациями и SHA256, `split.json` с разделением по содержимому и `review.csv` с ручными полями. По умолчанию в holdout пять уникальных видео каждого класса. Ручные поля CSV сохраняются при повторном аудите.

Просмотрщик показывает development, позволяет добавить несколько событий, экспортировать JSON и загрузить его обратно для продолжения работы. Импорт проверяет интервалы, временные точки и соответствие видео; ошибочный файл не заменяет текущую разметку. JSON и ручной CSV пока отдельные инструменты: автоматического слияния между ними нет. Holdout оставьте для итоговой проверки. Точка исходной разметки не превращается в интервал автоматически. Разные точки у одинакового ролика сохраняются для ручной проверки.

## Проверенные правила

`config/rules.json` изначально содержит `[]`. Заполняйте его после просмотра и проверки источников. См. [формат правил](docs/rules.md). Проверяются только активные условия из реестра, а не все возможные нарушения. Нельзя определять невидимые операции в АБИС, намерения или личность людей.

## Модель и анализ

```powershell
# Эта команда намеренно использует сеть только для подготовки весов
.venv/Scripts/python.exe -m cashlens_ai.cli download-model --output local/models/qwen3-vl-2b
# Проверка на синтетических изображениях, без банковских видео
.venv/Scripts/python.exe scripts/smoke_model.py --model-dir local/models/qwen3-vl-2b
# Требует заполненного проверенного реестра
.venv/Scripts/python.exe -m cashlens_ai.cli analyze 'C:/dataset/videos/video_001.mov' --model-dir local/models/qwen3-vl-2b --rules config/rules.json --output local/runs/job-001
```

Каждому заданию нужен новый каталог. Там сохраняются `result.json` и JPEG-доказательства. При ошибке или тайм-ауте завершённый результат не выдаётся. Кадры и описания не передаются внешним AI.

Начальные настройки: 1 кадр/с, окна по 8 кадров с перекрытием; уточнение найденных участков с большей частотой и разрешением. Это параметры для измерения, не обещание качества. Короткие действия могут пропускаться, мелкие/перекрытые детали — теряться. Метаданные поворота пока не применяются. Близкие события одного типа могут объединяться; порог задаёт `AnalyzerConfig.merge_sec`.

Внутренний бюджет — 240 секунд. Проверки и Transformers `max_time` кооперативные: отдельный вызов может закончиться позже. Серверу нужен жёсткий watchdog процесса, запас до 300 секунд и контроль очереди. Модель загрузите до приёма заданий и повторно используйте в одном GPU worker.

## Оценка

```powershell
.venv/Scripts/python.exe -m cashlens_ai.cli evaluate --manifest local/audit/manifest.json --predictions local/predictions.json --split holdout --output local/evaluation.json
```

Предсказания — JSON-массив `[{"sha256":"...","status":"done","result":{...AnalysisResult...}}]`. Сбой обозначается `status: "failed"`. Пропуски и ошибки не становятся отрицательными предсказаниями. Смещение к слабым точкам разметки — описательная величина, не официальная метрика локализации. Типы без ручной разметки оценивать нельзя.

## Локальные файлы

`local/`, видео, таблицы, изображения, веса и секреты исключены из Git. Не добавляйте их через `git add -f`. Публично сохраняются код, синтетические тесты и общая документация.

Компоненты: [Qwen3-VL-2B Apache 2.0](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [PyTorch CUDA](https://pytorch.org/get-started/previous-versions/), [bitsandbytes Windows](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation).
