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
const validJobId = (value) => typeof value === "string" && /^[a-f0-9]{32}$/.test(value);

function rememberJob(jobId) {
  const url = new URL(window.location.href);
  if (validJobId(jobId)) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  window.history.replaceState(null, "", url);
}

function restoreJob() {
  const jobId = new URL(window.location.href).searchParams.get("job");
  if (jobId === null) return;
  if (!validJobId(jobId)) {
    rememberJob(null);
    $("status").textContent = "Ссылка содержит некорректный идентификатор задачи. Выберите видео для нового анализа.";
    return;
  }
  busy = true;
  const run = ++activeRun;
  syncControls();
  $("status").textContent = "Восстанавливаем результат ранее загруженного видео…";
  // Existing jobs can be read even while the model is loading after a restart.
  poll(jobId, Date.now(), run, 0);
}

function syncControls() {
  fileInput.disabled = !ready || busy;
  drop.setAttribute("aria-disabled", String(fileInput.disabled));
  $("submit").disabled = !ready || busy || !fileInput.files[0];
  $("submit").textContent = analyzerName === "mock" ? "Запустить демонстрацию" : "Анализировать";
}

function setFile(file) {
  if (!file || !ready || busy) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  $("drop-text").textContent = `${file.name} — ${(file.size / 1048576).toFixed(1)} МБ`;
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

async function getJson(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const res = await fetch(url, { cache: "no-store", signal: controller.signal });
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
  link.textContent = href || "Ссылка на источник недоступна";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  if (href) link.href = href;
}

function renderCoverage(health) {
  const signature = JSON.stringify([health.rules, health.limitations]);
  if (signature === coverageSignature) return;
  coverageSignature = signature;
  const rules = Array.isArray(health.rules) ? health.rules : [];
  $("coverage-summary").textContent = `Подключённые правила (${rules.length}) и ограничения`;
  $("limitations").textContent = typeof health.limitations === "string" && health.limitations
    ? health.limitations
    : "Сервис ищет видимые признаки по подключённым правилам. Результат требует проверки специалистом и не является окончательным юридическим заключением.";
  const list = $("rules");
  list.replaceChildren();
  if (!rules.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Правила пока не загружены. Анализ возможен после подготовки модели и реестра правил.";
    list.append(empty);
  }
  rules.forEach((rule) => {
    const item = document.createElement("div");
    item.className = "coverage-rule";
    const title = document.createElement("strong");
    title.textContent = rule.title || rule.id;
    const condition = document.createElement("p");
    condition.textContent = rule.observable_condition || "";
    const source = document.createElement("p");
    source.textContent = [rule.source?.document, rule.source?.section].filter(Boolean).join(", ");
    const link = document.createElement("a");
    setSourceLink(link, rule.source);
    item.append(title, condition, source, link);
    list.append(item);
  });
}

function renderPerformance(health) {
  const info = health.performance;
  const visible = health.analyzer === "model" && info && typeof info.profile === "string";
  $("performance").hidden = !visible;
  $("performance-note").hidden = !visible;
  if (!visible) return;
  const profile = info.profile === "balanced" ? "сбалансированный" : "детальные кадры";
  const duty = Math.round(Number(info.gpu_duty_cycle) * 100);
  const memory = Math.round(Number(info.gpu_memory_fraction) * 100);
  $("performance").textContent = `Профиль: ${profile} · целевая доля работы GPU: ${duty}% · лимит памяти PyTorch: ${memory}%`;
  $("performance-note").textContent = (duty < 100
    ? "Между вычислениями есть паузы; краткие пики GPU до 100% возможны. "
    : "Дополнительные паузы GPU отключены; загрузка может достигать 100%. ")
    + (info.profile === "balanced"
      ? `Уменьшенные кадры и выборка до ${info.max_refine_frames} кадров на уточнение могут пропустить мелкие объекты и короткие действия.`
      : "Более крупные кадры требуют больше памяти и времени; качество распознавания всё равно требует проверки.");
}

async function refreshHealth() {
  try {
    const health = await getJson("/health");
    analyzerName = health.analyzer;
    ready = health.analyzer_ready === true && health.analyzer_state === "ready";
    const banner = $("model-status");
    if (ready && analyzerName === "mock") {
      banner.className = "model-status mock";
      banner.textContent = "ДЕМО-РЕЖИМ: результат создаёт заглушка. Содержимое видео не анализируется.";
    } else if (ready) {
      banner.className = "model-status ready";
      banner.textContent = "Локальная модель готова к анализу";
    } else if (health.analyzer_state === "failed" || health.analyzer_error) {
      banner.className = "model-status failed";
      banner.textContent = `Модель не готова: ${health.analyzer_error || "не удалось загрузить модель"}`;
    } else if (health.analyzer_state === "stopped") {
      banner.className = "model-status failed";
      banner.textContent = "Анализатор остановлен. Для работы требуется перезапуск локального сервиса.";
    } else {
      banner.className = "model-status loading";
      banner.textContent = "Подготавливаем локальную модель… После загрузки станет доступен выбор видео.";
    }
    $("model-name").textContent = health.model_name ? `Модель: ${health.model_name}` : "";
    renderPerformance(health);
    renderCoverage(health);
  } catch {
    ready = false;
    $("model-status").className = "model-status failed";
    $("model-status").textContent = "Нет связи с локальным сервисом. Проверка подключения продолжится автоматически.";
  } finally {
    syncControls();
    setTimeout(refreshHealth, ready ? 15000 : 4000);
  }
}

function resetProgress() {
  $("progress").hidden = true;
  $("progress").value = 0;
}

function failRun(message, run) {
  if (run !== activeRun) return;
  clearTimeout(pollTimer);
  busy = false;
  resetProgress();
  $("status").textContent = message;
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
  $("status").textContent = "Загрузка видео на локальный сервер…";
  xhr.upload.onprogress = (ev) => {
    if (run !== activeRun) return;
    if (ev.lengthComputable) $("progress").value = (ev.loaded / ev.total) * 100;
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
    $("status").textContent = "Видео загружено. Ожидаем анализ…";
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
      if (!Array.isArray(job.violations) || !["violation", "no_violation"].includes(job.verdict)) {
        throw new Error("Сервер вернул неполный результат анализа");
      }
      render(job);
      busy = false;
      resetProgress();
      $("status").textContent = `Готово за ${Number(job.processing_time_sec || secs).toFixed(1)} с`;
      syncControls();
      return;
    }
    if (job.status === "failed") {
      rememberJob(null);
      failRun(`Не удалось обработать видео: ${job.error || "неизвестная ошибка анализатора"}`, run);
      return;
    }
    if (!["processing", "queued"].includes(job.status)) throw new Error("Неизвестный статус задачи");
    $("status").textContent = job.stage === "queued" || job.status === "queued"
      ? `В очереди (перед вами: ${job.queue_position ?? 0}) · ${secs} с`
      : `Локальный анализ видео… ${secs} с`;
    pollTimer = setTimeout(() => poll(jobId, started, run, 0), 2000);
  } catch (error) {
    if (run !== activeRun) return;
    if (failures >= 2) {
      failRun(`Не удалось получить результат: ${error.message}. Задача: ${jobId}.`, run);
      return;
    }
    $("status").textContent = "Связь с сервером прервалась. Повторяем запрос результата…";
    pollTimer = setTimeout(() => poll(jobId, started, run, failures + 1), 3000);
  }
}

function fmt(sec) {
  const ticks = Math.round(Math.max(0, Number(sec) || 0) * 10);
  return `${String(Math.floor(ticks / 600)).padStart(2, "0")}:${String(Math.floor(ticks / 10) % 60).padStart(2, "0")}.${ticks % 10}`;
}

function seek(sec) {
  if (!Number.isFinite(Number(sec))) return;
  const player = $("player");
  player.currentTime = Math.max(0, Number(sec));
  player.scrollIntoView({ behavior: "smooth", block: "center" });
}

function render(job) {
  const ok = job.verdict === "no_violation";
  const items = job.violations;
  $("demo").hidden = job.analyzer !== "mock";
  $("result-model").textContent = job.analyzer === "mock"
    ? "Демонстрационный результат. Модель не анализировала видео."
    : `Видео обработано локальной моделью${$("model-name").textContent ? ` · ${$("model-name").textContent}` : ""}`;
  const verdict = $("verdict");
  verdict.className = `verdict ${ok ? "ok" : "bad"}`;
  verdict.textContent = ok
    ? "Признаки нарушений по подключённым правилам не обнаружены"
    : `Обнаружены признаки возможных нарушений: ${items.length}`;

  const player = $("player");
  const videoUrl = mediaUrl(job.video_url);
  const originalUrl = mediaUrl(job.original_url);
  player.onerror = () => {
    if (originalUrl && player.src !== originalUrl) player.src = originalUrl;
  };
  if (videoUrl || originalUrl) player.src = videoUrl || originalUrl;
  else player.removeAttribute("src");

  const timeline = $("timeline");
  timeline.replaceChildren();
  const duration = Number(job.duration_sec) || 1;
  items.forEach((ev) => {
    const bar = document.createElement("span");
    bar.style.left = `calc(${Math.min(100, Math.max(0, (Number(ev.timecode_sec) || 0) / duration * 100))}% - 3px)`;
    bar.title = `${fmt(ev.timecode_sec)} — ${ev.type}`;
    bar.onclick = () => seek(ev.timecode_sec);
    timeline.append(bar);
  });

  const list = $("events");
  list.replaceChildren();
  const tpl = $("event-tpl");
  items.forEach((ev) => {
    const node = tpl.content.cloneNode(true);
    const tc = node.querySelector(".timecode");
    tc.textContent = fmt(ev.timecode_sec);
    tc.onclick = () => seek(ev.timecode_sec);
    node.querySelector(".type").textContent = ev.type;
    node.querySelector(".description").textContent = ev.description;
    const frames = node.querySelector(".frames");
    (Array.isArray(ev.evidence_frames) ? ev.evidence_frames : []).forEach((frame) => {
      const url = mediaUrl(frame.url);
      if (!url) return;
      const fig = document.createElement("figure");
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      const img = document.createElement("img");
      img.src = url;
      img.alt = `Кадр-доказательство ${fmt(frame.timestamp_sec)}`;
      img.loading = "lazy";
      link.append(img);
      const cap = document.createElement("figcaption");
      cap.textContent = fmt(frame.timestamp_sec);
      fig.append(link, cap);
      frames.append(fig);
    });
    node.querySelector(".doc").textContent = ev.source?.document || "Источник не указан";
    node.querySelector(".section").textContent = ev.source?.section || "";
    setSourceLink(node.querySelector(".source"), ev.source);
    list.append(node);
  });

  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
  downloadUrl = URL.createObjectURL(blob);
  $("download").href = downloadUrl;
  $("result").hidden = false;
}

syncControls();
refreshHealth();
restoreJob();
