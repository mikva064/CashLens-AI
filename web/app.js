const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const drop = $("drop");
let pollTimer = null;

function setFile(file) {
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  $("drop-text").textContent = `${file.name} — ${(file.size / 1048576).toFixed(1)} МБ`;
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

function upload(file) {
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/web/analyze");
  $("progress").hidden = false;
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("progress").value = (e.loaded / e.total) * 100;
    $("status").textContent = "Загрузка видео…";
  };
  xhr.onload = () => {
    $("progress").hidden = true;
    const body = safeJson(xhr.responseText);
    if (xhr.status !== 202) {
      $("status").textContent = `Ошибка: ${body?.detail || xhr.status}`;
      $("submit").disabled = false;
      return;
    }
    poll(body.job_id);
  };
  xhr.onerror = () => {
    $("status").textContent = "Сеть недоступна";
    $("submit").disabled = false;
  };
  xhr.send(form);
}

function safeJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

function poll(jobId) {
  const started = Date.now();
  const tick = async () => {
    const res = await fetch(`/web/jobs/${jobId}`);
    const job = await res.json();
    const secs = Math.round((Date.now() - started) / 1000);
    if (job.status === "queued") {
      $("status").textContent = `В очереди (перед вами: ${job.queue_position}) · ${secs} с`;
    } else if (job.status === "processing") {
      $("status").textContent = `Анализ видео… ${secs} с`;
    } else {
      clearInterval(pollTimer);
      $("submit").disabled = false;
      if (job.status === "done") {
        $("status").textContent = `Готово за ${job.processing_time_sec} с`;
        render(job);
      } else {
        $("status").textContent = `Не удалось обработать: ${job.error}`;
      }
    }
  };
  tick();
  pollTimer = setInterval(tick, 2000);
}

function seek(sec) {
  const player = $("player");
  player.currentTime = sec;
  player.scrollIntoView({ behavior: "smooth", block: "center" });
}

function render(job) {
  const ok = job.verdict === "no_violation";
  const verdict = $("verdict");
  verdict.className = `verdict ${ok ? "ok" : "bad"}`;
  verdict.textContent = ok
    ? "Нарушение не обнаружено"
    : `Обнаружены признаки возможного нарушения: ${job.events.length}`;

  const player = $("player");
  player.src = job.video_url || "";

  const timeline = $("timeline");
  timeline.replaceChildren();
  const duration = job.duration_sec || 1;
  job.events.forEach((ev) => {
    const bar = document.createElement("span");
    bar.style.left = `${(ev.start_sec / duration) * 100}%`;
    bar.style.width = `${((ev.end_sec - ev.start_sec) / duration) * 100}%`;
    bar.title = `${ev.timecode} — ${ev.type_title}`;
    bar.onclick = () => seek(ev.start_sec);
    timeline.append(bar);
  });

  const list = $("events");
  list.replaceChildren();
  const tpl = $("event-tpl");
  job.events.forEach((ev) => {
    const node = tpl.content.cloneNode(true);
    const tc = node.querySelector(".timecode");
    tc.textContent = ev.timecode;
    tc.onclick = () => seek(ev.start_sec);
    node.querySelector(".type").textContent = ev.type_title;
    if (ev.confidence != null) {
      node.querySelector(".confidence").textContent = `уверенность ${Math.round(ev.confidence * 100)}%`;
    }
    node.querySelector(".description").textContent = ev.description;
    const frames = node.querySelector(".frames");
    ev.evidence_frames.forEach((f) => {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = f.url;
      img.alt = `Кадр ${f.sec} с`;
      img.onclick = () => window.open(f.url, "_blank");
      const cap = document.createElement("figcaption");
      cap.textContent = `${f.sec.toFixed(1)} с`;
      fig.append(img, cap);
      frames.append(fig);
    });
    node.querySelector(".requirement").textContent = ev.rule.requirement || "—";
    const src = node.querySelector(".source");
    src.textContent = ev.rule.source;
    if (ev.rule.url) src.href = ev.rule.url; else src.removeAttribute("href");
    node.querySelector(".unverified").hidden = ev.rule.verified;
    list.append(node);
  });

  const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
  $("download").href = URL.createObjectURL(blob);
  $("result").hidden = false;
}
