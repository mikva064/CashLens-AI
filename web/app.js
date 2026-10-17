const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const drop = $("drop");
let pollTimer = null;

/* ---------- Режим и список правил ---------- */

async function loadRules() {
  try {
    const info = await (await fetch("/web/rules")).json();
    const mode = $("mode");
    mode.hidden = false;
    if (info.analyzer === "model" && info.ready) {
      mode.textContent = "Модель работает локально";
      mode.classList.add("live");
    } else if (info.analyzer === "model") {
      mode.textContent = "Модель не загружена";
    } else {
      mode.textContent = "Демо-режим";
    }
    const list = $("rules");
    list.replaceChildren();
    if (!info.rules.length) {
      const li = document.createElement("li");
      li.textContent = "Список проверяемых правил ещё не заполнен.";
      list.append(li);
      return;
    }
    info.rules.forEach((r) => {
      const li = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = r.title;
      const cond = document.createElement("span");
      cond.textContent = r.condition;
      const src = document.createElement("span");
      src.textContent = r.source ? `${r.source.document}, ${r.source.section}` : "";
      li.append(title, cond, src);
      list.append(li);
    });
  } catch {
    /* интерфейс работает и без этого блока */
  }
}
loadRules();

/* ---------- Выбор файла ---------- */

function setFile(file) {
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  $("drop-title").textContent = file.name;
  $("drop-hint").textContent = `${(file.size / 1048576).toFixed(1)} МБ, готово к проверке`;
  drop.classList.add("ready");
  $("submit").disabled = false;
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => setFile(ev.dataTransfer.files[0]));

$("upload").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  clearInterval(pollTimer);
  $("result").hidden = true;
  $("submit").disabled = true;
  upload(file);
});

/* ---------- Этапы ---------- */

const ORDER = ["upload", "queued", "processing", "done"];

function setStep(step, text, isError = false) {
  const steps = $("steps");
  steps.hidden = false;
  const idx = ORDER.indexOf(step);
  steps.querySelectorAll("li").forEach((li, i) => {
    li.className = i < idx || step === "done" ? "done" : i === idx ? "now" : "";
  });
  const status = $("status");
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function fail(text) {
  setStep("upload", text, true);
  $("submit").disabled = false;
}

function upload(file) {
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/web/analyze");
  xhr.upload.onprogress = (e) => {
    const pct = e.lengthComputable ? Math.round((e.loaded / e.total) * 100) : 0;
    setStep("upload", `Загружаем запись: ${pct}%`);
  };
  xhr.onload = () => {
    let body = null;
    try { body = JSON.parse(xhr.responseText); } catch { /* не JSON */ }
    if (xhr.status !== 200) {
      fail(`Не удалось загрузить: ${body?.detail || `ошибка сервера ${xhr.status}`}. Проверьте файл и попробуйте снова.`);
      return;
    }
    poll(body.job_id);
  };
  xhr.onerror = () => fail("Нет связи с сервером. Проверьте подключение и попробуйте снова.");
  setStep("upload", "Загружаем запись: 0%");
  xhr.send(form);
}

function poll(jobId) {
  // Ссылка на результат: адрес с #job=… можно переслать коллеге.
  history.replaceState(null, "", `#job=${jobId}`);
  const started = Date.now();
  const tick = async () => {
    let job;
    try {
      job = await (await fetch(`/web/jobs/${jobId}`)).json();
    } catch {
      return; // временный сбой сети — попробуем на следующем шаге
    }
    const secs = Math.round((Date.now() - started) / 1000);
    if (job.stage === "queued") {
      setStep("queued", `Перед вами в очереди: ${job.queue_position}. Прошло ${secs} с.`);
    } else if (job.status === "processing") {
      setStep("processing", `Модель просматривает запись. Прошло ${secs} с, лимит — 5 минут.`);
    } else {
      clearInterval(pollTimer);
      $("submit").disabled = false;
      if (job.status === "done") {
        setStep("done", "Проверка завершена.");
        render(job);
      } else {
        fail(`Проверка не завершилась: ${job.error}`);
      }
    }
  };
  tick();
  pollTimer = setInterval(tick, 2000);
}

/* ---------- Результат ---------- */

function clock(sec) {
  const s = Math.floor(sec);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

function seek(sec) {
  const player = $("player");
  player.currentTime = sec;
  player.pause();
  if (window.matchMedia("(max-width: 760px)").matches) {
    player.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function zoom(src, caption) {
  $("zoom-img").src = src;
  $("zoom-img").alt = caption;
  $("zoom-cap").textContent = caption;
  $("zoom").showModal();
}

function render(job) {
  const items = job.violations;
  const found = items.length > 0;
  $("demo").hidden = job.analyzer !== "mock";

  const verdict = $("verdict");
  verdict.classList.toggle("clear", !found);
  $("verdict-title").textContent = found
    ? "Обнаружены признаки возможных нарушений"
    : "Нарушение не обнаружено";
  const count = found ? `${items.length} ${plural(items.length, "событие", "события", "событий")}. ` : "";
  $("verdict-meta").textContent = `${count}Проверка заняла ${Math.round(job.processing_time_sec)} с.`;
  verdict.classList.remove("reveal");
  void verdict.offsetWidth; // перезапуск анимации при повторной проверке
  verdict.classList.add("reveal");

  // Видео и шкала
  const player = $("player");
  player.onerror = () => {
    if (!player.src.endsWith(job.original_url)) player.src = job.original_url;
  };
  player.src = job.video_url;
  const duration = job.duration_sec || 1;
  $("track-end").textContent = clock(duration);
  const track = $("track");
  track.replaceChildren();
  const head = document.createElement("span");
  head.className = "head";
  track.append(head);
  player.ontimeupdate = () => { head.style.left = `${(player.currentTime / duration) * 100}%`; };
  items.forEach((ev) => {
    const mark = document.createElement("button");
    mark.type = "button";
    mark.style.left = `${(ev.timecode_sec / duration) * 100}%`;
    mark.title = `${clock(ev.timecode_sec)} ${ev.type}`;
    mark.setAttribute("aria-label", `Перейти к ${clock(ev.timecode_sec)}: ${ev.type}`);
    mark.onclick = () => seek(ev.timecode_sec);
    track.append(mark);
  });

  // События
  const list = $("events");
  list.replaceChildren();
  const tpl = $("event-tpl");
  items.forEach((ev, index) => {
    const node = tpl.content.cloneNode(true);
    const tc = node.querySelector(".timecode");
    const tenths = Math.floor((ev.timecode_sec % 1) * 10);
    const small = document.createElement("small");
    small.textContent = `.${tenths}`;
    tc.replaceChildren(clock(ev.timecode_sec), small);
    tc.onclick = () => seek(ev.timecode_sec);
    node.querySelector(".event-type").textContent = ev.type;
    node.querySelector(".event-desc").textContent = ev.description;

    const frames = node.querySelector(".frames");
    ev.evidence_frames.forEach((f, i) => {
      const fig = document.createElement("figure");
      const btn = document.createElement("button");
      btn.type = "button";
      const img = document.createElement("img");
      img.src = f.url;
      const caption = i === 0
        ? `${f.timestamp_sec.toFixed(1)} с, момент события`
        : `${f.timestamp_sec.toFixed(1)} с`;
      img.alt = `Кадр на ${caption}`;
      btn.append(img);
      btn.onclick = () => zoom(f.url, `${ev.type}, кадр на ${caption}`);
      const cap = document.createElement("figcaption");
      cap.textContent = caption;
      fig.append(btn, cap);
      frames.append(fig);
    });

    node.querySelector(".basis-doc").textContent = `${ev.source.document}, ${ev.source.section}`;
    node.querySelector(".basis-link").href = ev.source.url;

    const review = node.querySelector(".review");
    const state = node.querySelector(".review-state");
    const buttons = review.querySelectorAll("button");
    const show = (decision) => {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.decision === decision)));
      state.textContent = decision === "confirmed" ? "Отмечено как нарушение"
        : decision === "rejected" ? "Отмечено как ложное срабатывание" : "";
    };
    show((job.feedback || {})[index]);
    buttons.forEach((b) => {
      b.onclick = async () => {
        const res = await fetch(`/web/jobs/${job.job_id}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ event_index: index, decision: b.dataset.decision }),
        });
        if (res.ok) show(b.dataset.decision);
        else state.textContent = "Отметка не сохранилась. Попробуйте ещё раз.";
      };
    });
    list.append(node);
  });

  $("clean").hidden = found;
  if (!found) $("checks").open = true;

  $("r-file").textContent = job.filename || "";
  $("r-date").textContent = new Date(job.created_at * 1000).toLocaleString("ru-RU");
  $("r-id").textContent = job.job_id;
  $("print").onclick = () => window.print();

  const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
  $("download").href = URL.createObjectURL(blob);
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

const linked = new URLSearchParams(location.hash.slice(1)).get("job");
if (linked) poll(linked);
