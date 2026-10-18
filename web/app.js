const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const drop = $("drop");
let pollTimer = null;
let busy = false;
let ready = false;
let activeRun = 0;
let downloadUrl = null;
let analyzerName = null;
let coverageSignature = null;
let ruleTitles = new Map();
let statsRows = [];
let statsLoaded = false;
let statsFetching = false;
let statsAgain = false;
const ORDER = ["upload", "queued", "processing", "done"];
const validJobId = (value) => typeof value === "string" && /^[a-f0-9]{32}$/.test(value);
const validSha256 = (value) => typeof value === "string" && /^[a-f0-9]{64}$/i.test(value);

function rememberJob(jobId) {
  const url = new URL(window.location.href);
  if (validJobId(jobId)) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  const fragment = new URLSearchParams(url.hash.slice(1));
  if (fragment.has("job")) {
    fragment.delete("job");
    url.hash = fragment.toString();
  }
  window.history.replaceState(null, "", url);
}

function restoreJob() {
  const url = new URL(window.location.href);
  const jobId = url.searchParams.get("job") ?? new URLSearchParams(url.hash.slice(1)).get("job");
  if (jobId === null) return;
  if (!validJobId(jobId)) {
    rememberJob(null);
    $("status").textContent = "Ссылка содержит некорректный идентификатор задачи. Выберите видео для нового анализа.";
    return;
  }
  busy = true;
  rememberJob(jobId);
  const run = ++activeRun;
  syncControls();
  setStep("processing", "Восстанавливаем результат ранее загруженного видео…");
  // Existing jobs can be read even while the model is loading after a restart.
  poll(jobId, Date.now(), run, 0);
}

function syncControls() {
  fileInput.disabled = !ready || busy;
  drop.setAttribute("aria-disabled", String(fileInput.disabled));
  $("submit").disabled = !ready || busy || !fileInput.files[0];
  $("submit").textContent = analyzerName === "mock" ? "Запустить демонстрацию" : "Проверить запись";
}

function setFile(file) {
  if (!file || !ready || busy) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  $("drop-title").textContent = file.name;
  $("drop-hint").textContent = `${(file.size / 1048576).toFixed(1)} МБ, готово к проверке`;
  drop.classList.add("ready");
  syncControls();
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((event) =>
  drop.addEventListener(event, (ev) => {
    ev.preventDefault();
    if (ready && !busy) drop.classList.add("over");
  }));
["dragleave", "drop"].forEach((event) =>
  drop.addEventListener(event, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => setFile(ev.dataTransfer.files[0]));

async function getJson(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const res = await fetch(url, { ...options, cache: "no-store", signal: controller.signal });
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new Error(typeof body?.detail === "string" ? body.detail : `HTTP ${res.status}`);
    if (!body || typeof body !== "object") throw new Error("Сервер вернул некорректный ответ");
    return body;
  } finally {
    clearTimeout(timeout);
  }
}

function sourceUrl(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}

function mediaUrl(value) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value, window.location.origin);
    return url.origin === window.location.origin && url.pathname.startsWith("/media/") ? url.href : null;
  } catch { return null; }
}

function setSourceLink(link, source) {
  const href = sourceUrl(source?.url);
  link.removeAttribute("href");
  link.textContent = href ? "Открыть текст нормы" : "Ссылка на источник недоступна";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  if (href) link.href = href;
}

function renderCoverage(health) {
  const signature = JSON.stringify([health.rules, health.observation_types, health.limitations]);
  if (signature === coverageSignature) return;
  coverageSignature = signature;
  const rules = Array.isArray(health.rules) ? health.rules : [];
  const observationTypes = Array.isArray(health.observation_types) ? health.observation_types : [];
  ruleTitles = new Map(rules.map((rule) => [rule.id, rule.title || rule.id]));
  renderStats();
  $("checks-title").textContent = "Что проверяет сервис";
  $("limitations").textContent = typeof health.limitations === "string" && health.limitations
    ? health.limitations
    : "Сервис ищет видимые признаки по подключённым правилам. Результат требует проверки специалистом и не является окончательным юридическим заключением.";
  const list = $("rules");
  list.replaceChildren();
  if (!rules.length) {
    const empty = document.createElement("li");
    empty.className = "muted";
    empty.textContent = "Правила анализа пока недоступны.";
    list.append(empty);
  }
  rules.forEach((rule) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = rule.title || rule.id;
    const condition = document.createElement("span");
    condition.textContent = rule.observable_condition || "";
    const source = document.createElement("span");
    source.textContent = [rule.source?.document, rule.source?.section].filter(Boolean).join(", ");
    const link = document.createElement("a");
    setSourceLink(link, rule.source);
    item.append(title, condition, source, link);
    list.append(item);
  });
  const observations = $("observation-types");
  observations.replaceChildren();
  if (!observationTypes.length) {
    const empty = document.createElement("li");
    empty.className = "muted";
    empty.textContent = "Дополнительные типы наблюдаемых эпизодов пока не подключены.";
    observations.append(empty);
  }
  observationTypes.forEach((entry) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    const condition = document.createElement("span");
    title.textContent = typeof entry === "string" ? entry : entry.title || entry.id;
    condition.textContent = typeof entry === "string" ? "Требует оценки человеком."
      : entry.observable_condition || entry.description || "Требует оценки человеком.";
    item.append(title, condition);
    observations.append(item);
  });
}

function renderStats() {
  if (!statsLoaded) return;
  const confirmed = statsRows.reduce((sum, row) => sum + row.confirmed, 0);
  const rejected = statsRows.reduce((sum, row) => sum + row.rejected, 0);
  $("stats-status").textContent = confirmed + rejected === 0 ? "Отметок пока нет."
    : `Всего отметок: ${confirmed + rejected} · подтверждено: ${confirmed} · ложных срабатываний: ${rejected}`;
  const list = $("stats-rules");
  list.replaceChildren();
  statsRows.forEach((row) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = ruleTitles.get(row.rule_id) || row.rule_id;
    const counts = document.createElement("span");
    counts.textContent = `Подтверждено: ${row.confirmed} · ложных срабатываний: ${row.rejected}`;
    item.append(title, counts);
    list.append(item);
  });
}

async function refreshStats() {
  if (statsFetching) { statsAgain = true; return; }
  statsFetching = true;
  try {
    const info = await getJson("/web/stats");
    if (!Array.isArray(info.by_rule) || info.by_rule.some((row) =>
      !row || typeof row.rule_id !== "string" || !Number.isSafeInteger(row.confirmed)
      || row.confirmed < 0 || !Number.isSafeInteger(row.rejected) || row.rejected < 0)) {
      throw new Error("Invalid reviewer statistics");
    }
    statsRows = info.by_rule;
    statsLoaded = true;
    renderStats();
  } catch {
    $("stats-status").textContent = statsLoaded
      ? "Показаны ранее загруженные отметки. Обновление временно недоступно."
      : "Отметки временно недоступны.";
  } finally {
    statsFetching = false;
    if (statsAgain) { statsAgain = false; refreshStats(); }
  }
}

function renderPerformance() {
  for (const id of ["performance", "performance-note"]) {
    $(id).hidden = true;
    $(id).textContent = "";
  }
}

async function refreshHealth() {
  try {
    const health = await getJson("/health");
    analyzerName = health.analyzer;
    ready = health.analyzer_ready === true && health.analyzer_state === "ready";
    const mode = $("mode");
    mode.hidden = false;
    mode.classList.toggle("live", ready && analyzerName === "model");
    mode.textContent = analyzerName === "mock" ? "Демо-режим"
      : ready ? "Локальная обработка" : "Сервис не готов";
    const banner = $("model-status");
    if (ready && analyzerName === "mock") {
      banner.className = "model-status mock";
      banner.textContent = "ДЕМО-РЕЖИМ: результат создаёт заглушка. Содержимое видео не анализируется.";
    } else if (ready) {
      banner.className = "model-status ready";
      banner.textContent = "Сервис готов к анализу";
    } else if (health.analyzer_state === "failed" || health.analyzer_error) {
      banner.className = "model-status failed";
      banner.textContent = `Сервис не готов: ${health.analyzer_error || "повторите попытку позже"}`;
    } else if (health.analyzer_state === "stopped") {
      banner.className = "model-status failed";
      banner.textContent = "Сервис остановлен. Перезапустите приложение.";
    } else {
      banner.className = "model-status loading";
      banner.textContent = "Подготавливаем сервис… Скоро можно будет загрузить видео.";
    }
    $("model-name").textContent = "";
    renderPerformance(health);
    renderCoverage(health);
  } catch {
    ready = false;
    $("mode").classList.remove("live");
    $("mode").textContent = "Нет связи с сервисом";
    $("model-status").className = "model-status failed";
    $("model-status").textContent = "Нет связи с сервисом. Повторяем подключение…";
  } finally {
    syncControls();
    setTimeout(refreshHealth, ready ? 15000 : 4000);
  }
}

function resetProgress() {
  $("progress").hidden = true;
  $("progress").value = 0;
}

function setStep(step, text, isError = false) {
  const steps = $("steps");
  steps.hidden = false;
  const index = ORDER.indexOf(step);
  steps.querySelectorAll("li").forEach((item, i) => {
    item.className = i < index || step === "done" ? "done" : i === index ? "now" : "";
  });
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function failRun(message, run) {
  if (run !== activeRun) return;
  clearTimeout(pollTimer);
  busy = false;
  resetProgress();
  setStep("upload", message, true);
  syncControls();
}

$("upload").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const file = fileInput.files[0];
  if (!file || !ready || busy) return;
  const run = ++activeRun;
  clearTimeout(pollTimer);
  rememberJob(null);
  $("result").hidden = true;
  $("player").pause();
  if (downloadUrl) {
    URL.revokeObjectURL(downloadUrl);
    downloadUrl = null;
  }
  busy = true;
  syncControls();
  upload(file, run);
});

function upload(file, run) {
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/web/analyze");
  xhr.timeout = 180000;
  $("progress").value = 0;
  $("progress").hidden = false;
  setStep("upload", "Загрузка видео на локальный сервер…");
  xhr.upload.onprogress = (ev) => {
    if (run !== activeRun) return;
    if (ev.lengthComputable) {
      $("progress").value = (ev.loaded / ev.total) * 100;
      setStep("upload", `Загружаем запись: ${Math.round($("progress").value)}%`);
    }
  };
  xhr.onload = () => {
    if (run !== activeRun) return;
    resetProgress();
    let body;
    try { body = JSON.parse(xhr.responseText); } catch { body = null; }
    if (xhr.status < 200 || xhr.status >= 300) {
      const detail = typeof body?.detail === "string" ? body.detail : `HTTP ${xhr.status}`;
      failRun(`Не удалось загрузить видео: ${detail}`, run);
      return;
    }
    if (!validJobId(body?.job_id)) {
      failRun("Сервер не вернул идентификатор задачи. Попробуйте загрузить видео повторно.", run);
      return;
    }
    rememberJob(body.job_id);
    setStep("queued", "Видео загружено. Ожидаем анализ…");
    poll(body.job_id, Date.now(), run, 0);
  };
  xhr.onerror = () => failRun("Нет связи с локальным сервером. Проверьте его запуск и повторите загрузку.", run);
  xhr.ontimeout = () => failRun("Истекло время загрузки видео. Проверьте локальный сервер и повторите попытку.", run);
  xhr.onabort = () => failRun("Загрузка отменена.", run);
  xhr.send(form);
}

async function poll(jobId, started, run, failures) {
  if (run !== activeRun) return;
  try {
    const job = await getJson(`/web/jobs/${encodeURIComponent(jobId)}`);
    if (run !== activeRun) return;
    const secs = Math.round((Date.now() - started) / 1000);
    if (job.status === "done") {
      if (!Array.isArray(job.violations) || !["violation", "no_violation", "requires_review"].includes(job.verdict)
          || (job.observations !== undefined && !Array.isArray(job.observations))) {
        throw new Error("Сервер вернул неполный результат анализа");
      }
      render(job);
      busy = false;
      resetProgress();
      setStep("done", `Проверка завершена за ${Number(job.processing_time_sec || secs).toFixed(1)} с`);
      syncControls();
      return;
    }
    if (job.status === "failed") {
      rememberJob(null);
      failRun(`Не удалось обработать видео: ${job.error || "неизвестная ошибка анализатора"}`, run);
      return;
    }
    if (!["processing", "queued"].includes(job.status)) throw new Error("Неизвестный статус задачи");
    if (job.stage === "queued" || job.status === "queued") {
      setStep("queued", `Перед вами в очереди: ${job.queue_position ?? 0}. Прошло ${secs} с.`);
    } else {
      setStep("processing", `Модель просматривает запись. Прошло ${secs} с.`);
    }
    pollTimer = setTimeout(() => poll(jobId, started, run, 0), 2000);
  } catch (error) {
    if (run !== activeRun) return;
    if (failures >= 2) {
      failRun(`Не удалось получить результат: ${error.message}. Задача: ${jobId}.`, run);
      return;
    }
    setStep("processing", "Связь с сервером прервалась. Повторяем запрос результата…");
    pollTimer = setTimeout(() => poll(jobId, started, run, failures + 1), 3000);
  }
}

function fmt(sec) {
  const ticks = Math.round(Math.max(0, Number(sec) || 0) * 10);
  return `${String(Math.floor(ticks / 600)).padStart(2, "0")}:${String(Math.floor(ticks / 10) % 60).padStart(2, "0")}.${ticks % 10}`;
}

function clock(sec) {
  return fmt(sec).split(".")[0];
}

function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

function seek(sec) {
  if (!Number.isFinite(Number(sec))) return;
  const player = $("player");
  player.currentTime = Math.max(0, Number(sec));
  player.pause();
  if (window.matchMedia("(max-width: 760px)").matches) {
    player.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function zoom(src, caption) {
  const url = mediaUrl(src);
  if (!url) return;
  $("zoom-img").src = url;
  $("zoom-img").alt = caption;
  $("zoom-cap").textContent = caption;
  $("zoom").showModal();
}

function updateDownload(job) {
  if (downloadUrl) { URL.revokeObjectURL(downloadUrl); downloadUrl = null; }
  const artifactUrl = mediaUrl(job.result_json_url);
  const expectedUrl = validJobId(job.job_id)
    ? new URL(`/media/${job.job_id}/result.json`, window.location.origin).href : null;
  if (artifactUrl && artifactUrl === expectedUrl) {
    $("download").href = artifactUrl;
    $("download").textContent = "Скачать отчёт (JSON)";
    $("download-note").textContent = validSha256(job.result_sha256)
      ? ""
      : "Контрольная сумма отчёта недоступна.";
    return;
  }
  const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
  downloadUrl = URL.createObjectURL(blob);
  $("download").href = downloadUrl;
  $("download").textContent = "Экспортировать результат";
  $("download-note").textContent = "Архивный файл недоступен. Будет экспортирован текущий результат.";
}

function render(job) {
  const qualified = job.violations;
  const observations = Array.isArray(job.observations) ? job.observations : [];
  const found = qualified.length > 0;
  const reviewRequired = job.review_required === true || observations.length > 0 || job.verdict === "requires_review";
  const items = [
    ...qualified.map((event, index) => ({ event, index, observation: false })),
    ...observations.map((event) => ({ event, index: null, observation: true })),
  ].sort((left, right) => left.event.timecode_sec - right.event.timecode_sec);
  const runAtRender = activeRun;
  $("demo").hidden = job.analyzer !== "mock";
  $("result-model").textContent = job.analyzer === "mock"
    ? "Демонстрационный результат. Модель не анализировала видео."
    : "";
  $("result-model").hidden = job.analyzer !== "mock";

  const verdict = $("verdict");
  verdict.classList.toggle("clear", !found && !reviewRequired);
  verdict.classList.toggle("review-required", !found && reviewRequired);
  $("verdict-title").textContent = found ? "Обнаружены признаки возможных нарушений"
    : reviewRequired ? "Требуется проверка наблюдаемых эпизодов"
    : "Признаки нарушений по подключённым правилам не обнаружены";
  const counts = [];
  if (found) counts.push(`По нормативным правилам: ${qualified.length}`);
  if (observations.length) counts.push(`Эпизодов для проверки: ${observations.length}`);
  $("verdict-meta").textContent = `${counts.length ? `${counts.join(" · ")}. ` : ""}Проверка заняла ${Math.round(Number(job.processing_time_sec) || 0)} с.`;
  verdict.classList.remove("reveal");
  void verdict.offsetWidth;
  verdict.classList.add("reveal");

  const player = $("player");
  const videoUrl = mediaUrl(job.video_url);
  const originalUrl = mediaUrl(job.original_url);
  player.onerror = () => {
    if (originalUrl && player.src !== originalUrl) player.src = originalUrl;
  };
  if (videoUrl || originalUrl) player.src = videoUrl || originalUrl;
  else player.removeAttribute("src");
  const duration = Number(job.duration_sec) || 1;
  $("track-end").textContent = clock(duration);
  const track = $("track");
  track.replaceChildren();
  const head = document.createElement("span");
  head.className = "head";
  track.append(head);
  player.ontimeupdate = () => {
    head.style.left = `${Math.min(100, Math.max(0, player.currentTime / duration * 100))}%`;
  };
  items.forEach(({ event: ev, observation }) => {
    const mark = document.createElement("button");
    mark.type = "button";
    if (observation) mark.className = "observation";
    mark.style.left = `${Math.min(100, Math.max(0, (Number(ev.timecode_sec) || 0) / duration * 100))}%`;
    mark.title = `${fmt(ev.timecode_sec)} ${ev.type}${observation ? " — требует проверки" : ""}`;
    mark.setAttribute("aria-label", `Перейти к ${fmt(ev.timecode_sec)}: ${ev.type}${observation ? ", требует проверки" : ""}`);
    mark.onclick = () => seek(ev.timecode_sec);
    track.append(mark);
  });

  const list = $("events");
  list.replaceChildren();
  const observationList = $("observations");
  observationList.replaceChildren();
  $("qualified-section").hidden = !found;
  $("observations-section").hidden = observations.length === 0;
  $("track-legend").hidden = observations.length === 0;
  $("track-qualified").hidden = !found;
  const tpl = $("event-tpl");
  items.forEach(({ event: ev, index, observation }) => {
    const node = tpl.content.cloneNode(true);
    const tc = node.querySelector(".timecode");
    const formatted = fmt(ev.timecode_sec);
    const small = document.createElement("small");
    small.textContent = `.${formatted.split(".")[1]}`;
    tc.replaceChildren(formatted.split(".")[0], small);
    tc.onclick = () => seek(ev.timecode_sec);
    node.querySelector(".event-category-label").textContent = observation
      ? "Предполагаемая категория эпизода" : "Признак по нормативному правилу";
    node.querySelector(".event-type").textContent = ev.type;
    node.querySelector(".event-desc").textContent = ev.description;
    const commentaryLabels = {
      generated: "Автоматическое описание эпизода",
      uncertain: "Описание требует проверки",
      unavailable: "Описание эпизода недоступно",
    };
    const hasCommentaryStatus = Object.prototype.hasOwnProperty.call(commentaryLabels, ev.commentary_status);
    node.querySelector(".event-description-label").textContent = hasCommentaryStatus
      ? commentaryLabels[ev.commentary_status] : "Критерий выбранной категории";
    if (hasCommentaryStatus) {
      const category = node.querySelector(".category-description");
      category.hidden = false;
      node.querySelector(".category-description-text").textContent = ev.category_description || ev.type;
    } else {
      const notice = node.querySelector(".commentary-notice");
      notice.hidden = false;
      notice.textContent = "Описание эпизода недоступно.";
    }

    const frames = node.querySelector(".frames");
    (Array.isArray(ev.evidence_frames) ? ev.evidence_frames : []).forEach((frame) => {
      const url = mediaUrl(frame.url);
      if (!url) return;
      const fig = document.createElement("figure");
      const btn = document.createElement("button");
      btn.type = "button";
      const img = document.createElement("img");
      img.src = url;
      const frameTime = Number(frame.timestamp_sec), eventTime = Number(ev.timecode_sec);
      const offset = frameTime - eventTime;
      const tolerance = 0.001 + Number.EPSILON * Math.max(1, Math.abs(frameTime), Math.abs(eventTime));
      const role = Math.abs(offset) <= tolerance ? "момент события" : offset < 0 ? "до события" : "после события";
      const caption = `${fmt(frame.timestamp_sec)}, ${role}`;
      img.alt = `Кадр на ${caption}`;
      btn.append(img);
      btn.onclick = () => zoom(url, `${ev.type}, кадр на ${caption}`);
      const cap = document.createElement("figcaption");
      cap.textContent = caption;
      fig.append(btn, cap);
      frames.append(fig);
    });

    const clipUrl = mediaUrl(ev.clip_url);
    if (clipUrl) {
      node.querySelector(".event-clip").hidden = false;
      const clipPlayer = node.querySelector(".clip-player");
      clipPlayer.src = clipUrl;
      clipPlayer.onerror = () => { node.querySelector(".event-clip").hidden = true; };
    }

    if (observation) {
      node.querySelector(".event").classList.add("observation");
      node.querySelector(".observation-review").hidden = false;
      node.querySelector(".review-reason").textContent = ev.review_reason || "Для оценки эпизода требуется проверка человеком.";
      node.querySelector(".basis").remove();
      node.querySelector(".review").remove();
      observationList.append(node);
      return;
    }
    node.querySelector(".basis-doc").textContent = [ev.source?.document, ev.source?.section].filter(Boolean).join(", ") || "Источник не указан";
    setSourceLink(node.querySelector(".basis-link"), ev.source);

    const review = node.querySelector(".review");
    const state = node.querySelector(".review-state");
    const buttons = review.querySelectorAll("button");
    let saving = false;
    const show = (decision) => {
      buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.decision === decision)));
      state.textContent = decision === "confirmed" ? "Отмечено как нарушение"
        : decision === "rejected" ? "Отмечено как ложное срабатывание" : "";
    };
    show((job.feedback || {})[index]);
    buttons.forEach((button) => {
      button.disabled = !validJobId(job.job_id);
      button.onclick = async () => {
        if (saving || runAtRender !== activeRun || !validJobId(job.job_id)) return;
        saving = true;
        buttons.forEach((item) => { item.disabled = true; });
        state.textContent = "Сохраняем оценку…";
        try {
          const response = await getJson(`/web/jobs/${job.job_id}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ event_index: index, decision: button.dataset.decision }),
          });
          if (response.ok !== true) throw new Error("Feedback was not saved");
          if (runAtRender !== activeRun) return;
          job.feedback = { ...(job.feedback || {}), [index]: button.dataset.decision };
          show(button.dataset.decision);
          updateDownload(job);
          refreshStats();
        } catch {
          state.textContent = "Отметка не сохранилась. Попробуйте ещё раз.";
        } finally {
          saving = false;
          buttons.forEach((item) => { item.disabled = false; });
        }
      };
    });
    list.append(node);
  });

  $("clean").hidden = found || reviewRequired;
  if (!found && !reviewRequired) $("checks").open = true;
  $("r-file").textContent = job.filename || "Имя файла недоступно";
  const createdAt = Number(job.created_at);
  $("r-date").textContent = Number.isFinite(createdAt) && createdAt > 0
    ? new Date(createdAt * 1000).toLocaleString("ru-RU", { timeZone: "Asia/Qyzylorda" })
    : "Дата недоступна";
  $("r-id").textContent = job.job_id || "";
  $("r-input-sha256").textContent = validSha256(job.input_sha256) ? job.input_sha256 : "Не сохранён для этой проверки";
  $("r-result-sha256").textContent = validSha256(job.result_sha256) ? job.result_sha256 : "Не сохранён для этой проверки";
  $("print").onclick = () => window.print();
  updateDownload(job);
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

syncControls();
refreshHealth();
refreshStats();
restoreJob();
