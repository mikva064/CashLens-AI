"""Build an offline human annotation viewer; original videos stay in place."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .errors import DatasetValidationError


def _safe_json(value: Any) -> str:
    """Keep data inert even if an annotation contains an HTML closing tag."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _basename(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise DatasetValidationError("Manifest filenames must be nonempty basenames.")
    return value


def _viewer_data(manifest: dict[str, Any], include_holdout: bool) -> dict[str, Any]:
    try:
        if manifest.get("schema_version") != 1:
            raise DatasetValidationError("Unsupported audit manifest schema_version; expected 1.")
        videos_dir = Path(manifest["source"]["videos_dir"])
        if not videos_dir.is_absolute():
            raise DatasetValidationError("Manifest source.videos_dir must be an absolute local path.")
        allowed = {"development", "holdout"} if include_holdout else {"development"}
        selected = [group for group in manifest["groups"] if group["split"] in allowed]
        annotations = manifest["annotations"]
        videos = []
        seen: set[str] = set()
        for group in selected:
            sha = group["sha256"]
            if not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
                raise DatasetValidationError("Manifest groups need lowercase SHA256 identifiers.")
            if sha in seen:
                raise DatasetValidationError(f"Manifest repeats video group {sha}.")
            seen.add(sha)
            filenames = [_basename(name) for name in group["filenames"]]
            if not filenames:
                raise DatasetValidationError(f"Video group {sha} has no source filenames.")
            if group["label"] not in (0, 1):
                raise DatasetValidationError(f"Video group {sha} has no binary label.")
            source_annotations = []
            for row in annotations:
                if row["sha256"] != sha:
                    continue
                point = row.get("timestamp_sec")
                if point is not None and (isinstance(point, bool) or not isinstance(point, (int, float))
                                          or not math.isfinite(point) or point < 0):
                    raise DatasetValidationError(f"Invalid weak timestamp for group {sha}.")
                source_annotations.append({
                    "source_row": row["source_row"],
                    "filename": _basename(row["filename"]),
                    "label_filename": str(row.get("label_filename", row["filename"])),
                    "timestamp_raw": str(row.get("timestamp_raw", "")),
                    "timestamp_sec": point,
                    "annotation_kind": row.get("annotation_kind", "video_label_only"),
                    "review_flags": row.get("review_flags", []),
                })
            if not source_annotations:
                raise DatasetValidationError(f"Video group {sha} has no source annotations.")
            videos.append({
                "sha256": sha,
                "filenames": filenames,
                "video_urls": [(videos_dir / name).as_uri() for name in filenames],
                "label": group["label"],
                "split": group["split"],
                "source_annotations": source_annotations,
                "review_flags": group.get("review_flags", []),
            })
        videos.sort(key=lambda video: (video["filenames"][0].casefold(), video["sha256"]))
        return {"schema_version": 1, "includes_holdout": include_holdout, "videos": videos}
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError(f"Malformed audit manifest: {exc}") from exc


def build_review(
    manifest_path: Path, output_path: Path, *, include_holdout: bool = False,
) -> Path:
    """Generate one self-contained HTML file with development videos by default.

    Media use original file:// URLs and are never copied or uploaded. Holdout
    filenames, URLs, hashes and annotations are absent from the default payload.
    Opening the viewer and downloading JSON require no server or external assets.
    """
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot read audit manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetValidationError("Audit manifest must be a JSON object.")
    payload = _safe_json(_viewer_data(manifest, include_holdout))
    html = _HTML.replace("__CASHLENS_DATA__", payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_path.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(html)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return output_path.resolve()


_HTML = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src file: blob:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>CashLens · разметка видео</title>
<style>
:root{color-scheme:dark;font:15px/1.5 system-ui,sans-serif;background:#11151e;color:#e6edf7}
*{box-sizing:border-box}body{margin:0}button,input,select,textarea{font:inherit}button,select{cursor:pointer}
button,a{touch-action:manipulation}button{border:1px solid #46516a;background:#26324a;color:inherit;border-radius:7px;padding:7px 12px}
button:hover{background:#354566}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible,a:focus-visible{outline:2px solid #8dc3ff;outline-offset:2px}
header{padding:18px 24px;border-bottom:1px solid #344057;display:flex;gap:20px;align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{font-size:23px;margin:0}h2{font-size:20px;margin:0 0 12px}h3{font-size:16px;margin:0}p{margin:7px 0}small,.muted{color:#adb9cd}
.notice{padding:12px 24px;border-bottom:1px solid #534a2c;background:#282519;color:#f6dfa2}.notice p{margin:2px 0}
main{display:grid;grid-template-columns:270px minmax(0,1fr);min-height:70vh}aside{padding:18px;border-right:1px solid #344057;max-height:85vh;overflow:auto;position:sticky;top:0}
#video-list{display:grid;gap:8px;margin-top:12px}.video-item{text-align:left;width:100%;overflow-wrap:anywhere}.video-item[aria-current=true]{border-color:#9ac9ff;background:#314e70}.video-item small{display:block}
.workspace{padding:22px;min-width:0}.analysis-grid{display:grid;grid-template-columns:minmax(260px,440px) minmax(300px,1fr);gap:24px;align-items:start}
video{width:100%;max-height:65vh;background:#000;border:1px solid #344057;border-radius:9px}.toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:10px 0}.time{font-variant-numeric:tabular-nums}
.panel{background:#191f2c;border:1px solid #344057;border-radius:9px;padding:14px;margin:12px 0}.points{display:flex;gap:8px;flex-wrap:wrap}.points button{font-size:13px}.path-links{display:grid;gap:4px;overflow-wrap:anywhere}a{color:#9ac9ff}
.event{border:1px solid #46516a;border-radius:9px;padding:14px;margin:12px 0}.event-title{display:flex;justify-content:space-between;gap:10px;align-items:center}.event-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-top:12px}.wide{grid-column:1/-1}
label{display:grid;gap:4px}input,select,textarea{width:100%;border:1px solid #46516a;border-radius:6px;background:#0e1521;color:#e6edf7;padding:7px}textarea{min-height:70px;resize:vertical}.field-time{display:flex;gap:5px}.field-time button{font-size:12px;white-space:nowrap}
.danger{background:#422d35}.status{min-height:1.5em;color:#b2d9ff}#empty{padding:28px;border:1px dashed #46516a;border-radius:9px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.row label{display:flex;align-items:center;gap:8px}.row input[type=checkbox]{width:auto}#export{background:#315d47;border-color:#54896a}.error{color:#ffb5ad}.badge{font-size:12px;border:1px solid #59667b;padding:2px 7px;border-radius:99px}
@media(max-width:1050px){.analysis-grid{grid-template-columns:1fr}video{max-height:55vh}}@media(max-width:650px){main{grid-template-columns:1fr}aside{position:static;max-height:260px;border-right:0;border-bottom:1px solid #344057}.workspace{padding:16px}.event-grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
</head>
<body>
<header><div><h1>CashLens · разметка видео</h1><p class="muted" id="mode"></p></div><div><button id="export" type="button">Экспорт JSON</button><p id="save-status" class="muted" aria-live="polite">Изменений нет</p></div></header>
<div class="notice"><p><strong>Нормативные ссылки ещё не прошли правовую проверку.</strong> Поле rule_id заполняйте только после проверки источника.</p><p>Точки из Excel — ориентиры, а не границы событий. Разметьте интервалы вручную. Видео открываются с вашего диска. Перед закрытием экспортируйте JSON: изменения хранятся только в этой вкладке.</p></div>
<main><aside><h2>Уникальные видео</h2><p class="muted" id="progress"></p><div id="video-list"></div></aside>
<section class="workspace"><div id="empty" hidden>Нет видео для выбранного режима. Проверьте manifest.json и разделение данных.</div><div id="review-content">
<h2 id="video-title"></h2><div class="analysis-grid"><div>
<video id="player" controls preload="metadata"></video><div class="toolbar"><span id="clock" class="time">00:00.0</span><label for="speed">Скорость</label><select id="speed" style="width:auto"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></div>
<p id="media-status" class="muted" aria-live="polite"></p><div class="panel"><strong>Исходная разметка</strong><p id="source-label"></p><div id="points" class="points"></div><p id="flags" class="muted"></p></div>
<details class="panel"><summary>Исходные файлы и дубликаты</summary><p class="muted">Если браузер не воспроизводит MOV, откройте исходный файл в видеопроигрывателе. Все файлы группы имеют одинаковый SHA256.</p><div id="source-files" class="path-links"></div><p id="hash" class="muted"></p></details>
</div><div><div class="row"><h3>Ручная разметка</h3><span id="partition" class="badge"></span></div><div class="panel"><label>Заметки о видео<textarea id="video-notes" placeholder="Что видно, неоднозначности, качество записи"></textarea></label><p><label class="row"><input id="reviewed" type="checkbox" style="width:auto">Видео просмотрено полностью</label></p></div>
<div id="events"></div><button id="add-event" type="button">+ Добавить событие</button><p class="muted">У видео без нарушения список событий можно оставить пустым. Отметьте полный просмотр.</p><p id="validation" class="status" aria-live="polite"></p></div></div></div></section></main>
<template id="event-template"><article class="event"><div class="event-title"><h3 class="event-number"></h3><button class="delete-event danger" type="button">Удалить</button></div><div class="event-grid">
<label class="wide">Тип события<input data-field="type" type="text" placeholder="После просмотра выберите наблюдаемый тип"></label>
<label>Начало, секунды<div class="field-time"><input data-field="start_sec" type="number" min="0" step="0.1"><button data-current="start_sec" type="button">Сейчас</button></div></label>
<label>Конец, секунды<div class="field-time"><input data-field="end_sec" type="number" min="0" step="0.1"><button data-current="end_sec" type="button">Сейчас</button></div></label>
<label class="wide">rule_id · после проверки нормы<input data-field="rule_id" type="text" placeholder="Идентификатор проверенного правила"></label>
<label class="wide">Кадры-доказательства, секунды через запятую<div class="field-time"><input data-field="evidence_sec" type="text" placeholder="Например: 55.2, 57, 59.8"><button class="add-evidence" type="button">+ Сейчас</button></div></label>
<label class="wide">Описание наблюдаемого действия и сомнения<textarea data-field="notes"></textarea></label></div></article></template>
<script type="application/json" id="cashlens-data">__CASHLENS_DATA__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('cashlens-data').textContent);
const byId = id => document.getElementById(id);
const player = byId('player');
const state = data.videos.map(video => ({...video, reviewed: false, notes: '', events: []}));
let selected = -1;
let dirty = false;
let eventCounter = 0;
let pendingSeek = null;
function markDirty() { dirty = true; byId('save-status').textContent = 'Есть изменения · экспортируйте JSON'; }
function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, '0')}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
}
function updateProgress() {
  byId('progress').textContent = `Просмотрено ${state.filter(v => v.reviewed).length} из ${state.length}`;
  state.forEach((video, index) => {
    const button = byId(`video-${index}`);
    if (button) button.querySelector('small').textContent = `${video.label ? 'Нарушение по Excel' : 'Без нарушения по Excel'} · ${video.reviewed ? 'просмотрено' : 'ожидает просмотра'}`;
  });
}
function seek(seconds) {
  if (player.readyState < 1) { pendingSeek = seconds; return; }
  if (Number.isFinite(player.duration) && seconds > player.duration) {
    byId('media-status').textContent = 'Точка из разметки находится за пределами видео. Проверьте источник.';
    return;
  }
  player.currentTime = seconds;
}
function numeric(value) { return value.trim() === '' ? null : Number(value); }
function checkEvent(event) {
  if (event.start_sec !== null && (!Number.isFinite(event.start_sec) || event.start_sec < 0)) return 'Начало должно быть неотрицательным числом.';
  if (event.end_sec !== null && (!Number.isFinite(event.end_sec) || event.end_sec < 0)) return 'Конец должен быть неотрицательным числом.';
  if (event.start_sec !== null && event.end_sec !== null && event.end_sec < event.start_sec) return 'Конец события находится раньше начала.';
  if (event.evidence_sec.some(point => !Number.isFinite(point) || point < 0)) return 'Точки доказательств должны быть неотрицательными числами через запятую.';
  return '';
}
function showValidation() {
  const video = state[selected];
  const errors = video.events.map((event, index) => { const error = checkEvent(event); return error ? `Событие ${index + 1}: ${error}` : ''; }).filter(Boolean);
  byId('validation').textContent = errors.join(' ');
  byId('validation').classList.toggle('error', errors.length > 0);
}
function renderEvents() {
  const video = state[selected];
  const container = byId('events');
  container.replaceChildren();
  video.events.forEach((event, index) => {
    const article = byId('event-template').content.firstElementChild.cloneNode(true);
    article.querySelector('.event-number').textContent = `Событие ${index + 1}`;
    article.querySelectorAll('[data-field]').forEach(input => {
      const field = input.dataset.field;
      input.value = field === 'evidence_sec' ? event.evidence_input : (event[field] ?? '');
      input.addEventListener('input', () => {
        if (field === 'start_sec' || field === 'end_sec') event[field] = numeric(input.value);
        else if (field === 'evidence_sec') {
          event.evidence_input = input.value;
          event.evidence_sec = input.value.trim() ? input.value.split(/[,;]+/).map(part => part.trim() ? Number(part.trim()) : NaN) : [];
        } else event[field] = input.value;
        markDirty(); showValidation();
      });
    });
    article.querySelectorAll('[data-current]').forEach(button => button.addEventListener('click', () => {
      const field = button.dataset.current;
      const point = Math.round(player.currentTime * 10) / 10;
      event[field] = point;
      article.querySelector(`[data-field="${field}"]`).value = point;
      markDirty(); showValidation();
    }));
    article.querySelector('.add-evidence').addEventListener('click', () => {
      const point = Math.round(player.currentTime * 10) / 10;
      event.evidence_input = event.evidence_input.trim() ? `${event.evidence_input}, ${point}` : String(point);
      const input = article.querySelector('[data-field="evidence_sec"]');
      input.value = event.evidence_input;
      input.dispatchEvent(new Event('input'));
    });
    article.querySelector('.delete-event').addEventListener('click', () => {
      video.events.splice(index, 1); markDirty(); renderEvents();
    });
    container.append(article);
  });
  showValidation();
}
function selectVideo(index) {
  player.pause(); selected = index; pendingSeek = null;
  const video = state[index];
  document.querySelectorAll('.video-item').forEach((button, i) => button.setAttribute('aria-current', String(i === index)));
  byId('video-title').textContent = video.filenames.join(' / ');
  byId('source-label').textContent = video.label ? 'В Excel: нарушение. Тип и границы нужно уточнить.' : 'В Excel: без нарушения. Проверьте видео полностью.';
  byId('partition').textContent = video.split === 'holdout' ? 'Отложенная проверка' : 'Разработка';
  byId('flags').textContent = video.review_flags.includes('duplicate_timestamps_differ') ? 'У одинаковых видео разные временные точки. Проверьте обе: это могут быть разные события или неточность исходной разметки.' : (video.filenames.length > 1 ? 'Дубликаты объединены в одну группу.' : '');
  byId('points').replaceChildren();
  const points = video.source_annotations.filter(annotation => annotation.timestamp_sec !== null);
  if (!points.length) { const message = document.createElement('p'); message.textContent = 'Временных точек в исходной разметке нет.'; byId('points').append(message); }
  points.forEach(annotation => {
    const button = document.createElement('button'); button.type = 'button';
    button.textContent = `${formatTime(annotation.timestamp_sec)} · ${annotation.filename}`;
    button.title = `Строка ${annotation.source_row}: ${annotation.timestamp_raw}`;
    button.addEventListener('click', () => seek(annotation.timestamp_sec)); byId('points').append(button);
  });
  byId('source-files').replaceChildren();
  video.filenames.forEach((name, i) => {
    const link = document.createElement('a'); link.href = video.video_urls[i]; link.textContent = name; link.target = '_blank'; link.rel = 'noopener'; byId('source-files').append(link);
  });
  byId('hash').textContent = `SHA256: ${video.sha256}`;
  byId('video-notes').value = video.notes; byId('reviewed').checked = video.reviewed;
  byId('media-status').textContent = 'Загружается исходное видео…';
  byId('clock').textContent = '00:00.0';
  player.src = video.video_urls[0]; player.load(); renderEvents();
}
byId('mode').textContent = data.includes_holdout ? 'Разработка и отложенная выборка · режим итоговой проверки' : 'Только выборка для разработки · отложенные видео исключены';
state.forEach((video, index) => {
  const button = document.createElement('button'); button.type = 'button'; button.id = `video-${index}`; button.className = 'video-item'; button.setAttribute('aria-current', 'false');
  const name = document.createElement('span'); name.textContent = video.filenames.join(' / ');
  const status = document.createElement('small'); button.append(name, status);
  button.addEventListener('click', () => selectVideo(index)); byId('video-list').append(button);
});
byId('speed').addEventListener('change', event => { player.playbackRate = Number(event.target.value); });
player.addEventListener('loadedmetadata', () => {
  player.playbackRate = Number(byId('speed').value); byId('media-status').textContent = `Длительность ${formatTime(player.duration)}`;
  if (pendingSeek !== null) { const point = pendingSeek; pendingSeek = null; seek(point); }
});
player.addEventListener('timeupdate', () => { byId('clock').textContent = formatTime(player.currentTime); });
player.addEventListener('error', () => { byId('media-status').textContent = 'Не удалось воспроизвести файл. Проверьте путь или откройте MOV в отдельном видеопроигрывателе.'; });
byId('video-notes').addEventListener('input', event => { state[selected].notes = event.target.value; markDirty(); });
byId('reviewed').addEventListener('change', event => { state[selected].reviewed = event.target.checked; markDirty(); updateProgress(); });
byId('add-event').addEventListener('click', () => {
  state[selected].events.push({event_id: `event-${++eventCounter}`, type: '', start_sec: null, end_sec: null, rule_id: '', evidence_sec: [], evidence_input: '', notes: ''});
  markDirty(); renderEvents();
});
byId('export').addEventListener('click', () => {
  for (let i = 0; i < state.length; i++) {
    const invalid = state[i].events.findIndex(event => checkEvent(event));
    if (invalid !== -1) { selectVideo(i); byId('validation').textContent = `Исправьте событие ${invalid + 1} перед экспортом: ${checkEvent(state[i].events[invalid])}`; byId('validation').classList.add('error'); return; }
  }
  const exported = {schema_version: 1, purpose: 'manual_review', includes_holdout: data.includes_holdout, exported_at: new Date().toISOString(), videos: state.map(video => ({
    sha256: video.sha256, filenames: video.filenames, split: video.split, binary_label: video.label, source_annotations: video.source_annotations, reviewed: video.reviewed, notes: video.notes,
    events: video.events.map(({evidence_input, ...event}) => event)
  }))};
  const url = URL.createObjectURL(new Blob([JSON.stringify(exported, null, 2)], {type: 'application/json;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = `cashlens-review-${new Date().toISOString().slice(0,10)}.json`; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000); dirty = false; byId('save-status').textContent = 'JSON подготовлен · сохраните скачанный файл';
});
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
updateProgress();
if (state.length) selectVideo(0); else { byId('empty').hidden = false; byId('review-content').hidden = true; byId('export').disabled = true; }
</script>
</body>
</html>
'''
