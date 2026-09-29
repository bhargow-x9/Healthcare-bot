"use strict";

const $ = (sel) => document.querySelector(sel);
const NOT_FOUND_PREFIX = "Not found in our sources";
const KIND_LABELS = {
  prescription: "Your prescription",
  fda_label: "FDA drug label",
  medlineplus: "MedlinePlus",
  knowledge_base: "Reference book",
  glossary: "Glossary",
};

const state = { sessionId: null, sources: {}, file: null, tab: "file", audio: null, voiceQuestion: false };

// ---------- helpers ----------
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path, body, isForm = false) {
  const res = await fetch(path, {
    method: "POST",
    headers: isForm ? undefined : { "Content-Type": "application/json" },
    body: isForm ? body : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function rememberSources(list) {
  (list || []).forEach((s) => (state.sources[s.id] = s));
}

function chips(ids) {
  return (ids || [])
    .map((id) => {
      const s = state.sources[id];
      const tip = s ? `${s.title} – ${s.location}` : id;
      return `<button type="button" class="cite" data-src="${esc(id)}" title="${esc(tip)}">${esc(id)}</button>`;
    })
    .join("");
}

/** Escape text and turn inline [S3] / [S3, S5] markers into clickable chips. */
function withInlineCites(text) {
  return esc(text).replace(/\[((?:S\d+\s*,?\s*)+)\]/g, (_, group) => chips(group.match(/S\d+/g)));
}

function focusSource(id) {
  const el = document.getElementById(`src-${id}`);
  if (!el) return;
  el.open = true;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1600);
}

document.addEventListener("click", (e) => {
  const cite = e.target.closest(".cite");
  if (cite) focusSource(cite.dataset.src);
});

// ---------- status ----------
async function loadStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const pills = [
      [`Model: ${s.llm.provider} · ${s.llm.models.medical}`, s.llm.ok, s.llm.message],
      [`Knowledge base: ${s.knowledge_base.backend}`, s.knowledge_base.ready, s.knowledge_base.message],
      ["FDA / MedlinePlus", s.online_sources, s.online_sources ? "online sources on" : "disabled"],
    ];
    $("#status").innerHTML = pills
      .map(([label, ok, tip]) => `<span class="pill ${ok ? "ok" : "bad"}" title="${esc(tip)}">${esc(label)}</span>`)
      .join("");
  } catch {
    $("#status").innerHTML = `<span class="pill bad">Server not reachable</span>`;
  }
}

// ---------- input ----------
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    state.tab = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    $("#tab-file").classList.toggle("hidden", state.tab !== "file");
    $("#tab-text").classList.toggle("hidden", state.tab !== "text");
  })
);

function setFile(file) {
  state.file = file || null;
  $("#file-name").textContent = file ? `${file.name} (${Math.ceil(file.size / 1024)} KB)` : "";
}
$("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

function showError(msg) {
  $("#error").textContent = msg;
  $("#error").classList.toggle("hidden", !msg);
}

$("#analyze-btn").addEventListener("click", async () => {
  const form = new FormData();
  if (state.tab === "file") {
    if (!state.file) return showError("Please choose a photo, PDF or file first.");
    form.append("file", state.file);
  } else {
    const text = $("#rx-text").value.trim();
    if (!text) return showError("Please type or paste the prescription text.");
    form.append("text", text);
  }
  showError("");
  const btn = $("#analyze-btn");
  btn.disabled = true;
  $("#progress").classList.remove("hidden");
  $("#progress-text").textContent = "Uploading…";
  const started = Date.now();
  try {
    const { job_id } = await api("/api/analyze", form, true);
    const result = await waitForJob(job_id, started);
    state.sessionId = result.session_id;
    state.sources = {};
    rememberSources(result.sources);
    renderResults(result);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    $("#progress").classList.add("hidden");
  }
});

/** Poll the background analysis and show its real progress. */
async function waitForJob(jobId, started) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    const res = await fetch(`/api/analyze/${jobId}`);
    const job = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(job.error || "Lost track of the analysis. Please try again.");
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(job.error);
    const secs = Math.round((Date.now() - started) / 1000);
    const count = job.total ? ` (${job.done}/${job.total})` : "";
    $("#progress-text").textContent = `${job.message}${count} · ${secs}s`;
    $("#progress-bar").style.width = job.total ? `${Math.round((100 * job.done) / job.total)}%` : "5%";
  }
}

// ---------- results ----------
function field(label, f) {
  if (!f) return "";
  const missing = f.text.startsWith(NOT_FOUND_PREFIX);
  const flag = !missing && !f.verified
    ? `<div class="flag">⚠ Not confirmed by a source – check with your doctor or pharmacist.</div>` : "";
  return `<div class="field ${missing ? "missing" : ""}">
    <div class="field-label">${esc(label)}</div>
    <div class="field-text">${esc(f.text)} ${chips(f.sources)}</div>${flag}</div>`;
}

function medicineCard(m) {
  const d = m.details || {};
  const badges = [
    d.strength, d.form, d.dose,
    d.frequency_meaning || d.frequency_as_written, d.timing, d.duration && `for ${d.duration}`,
  ].filter(Boolean).map((b) => `<span class="badge">${esc(b)}</span>`).join("");
  const low = d.confidence === "low" ? `<span class="badge low">Hard to read – please double-check</span>` : "";
  const generic = d.generic_name && d.generic_name.toLowerCase() !== (d.brand_name || "").toLowerCase()
    ? ` · contains <strong>${esc(d.generic_name)}</strong>` : "";
  return `<article class="card item-card">
    <header><div>
      <h3>💊 ${esc(d.as_written || m.name)}</h3>
      <div class="item-sub">${esc(m.plain_name)}${generic}</div>
    </div></header>
    <div class="badges">${badges}${low}</div>
    ${field("What it is for", m.what_it_is_for)}
    ${field("How to take it (from your prescription)", m.how_to_take)}
    ${field("Common side effects", m.common_side_effects)}
    ${field("Important cautions", m.important_cautions)}
  </article>`;
}

function conditionCard(c) {
  return `<article class="card item-card">
    <header><div><h3>🩺 ${esc(c.name)}</h3><div class="item-sub">${esc(c.plain_name)}</div></div></header>
    ${field("What it is", c.what_it_is)}
    ${field("Common signs", c.common_signs)}
    ${field("What helps", c.what_helps)}
  </article>`;
}

function testCard(t) {
  return `<article class="card item-card">
    <header><div><h3>🧪 ${esc(t.name)}</h3><div class="item-sub">${esc(t.plain_name)}</div></div></header>
    ${field("What it checks", t.what_it_checks)}
    ${field("Why it matters", t.why_it_matters)}
    ${field("How to prepare", t.how_to_prepare)}
  </article>`;
}

function listCard(title, items, cls = "") {
  if (!items || !items.length) return "";
  return `<div class="card ${cls}"><h2>${title}</h2><ul class="plain">${items.map((i) => `<li>${i}</li>`).join("")}</ul></div>`;
}

function sourcesHtml() {
  const list = Object.values(state.sources).sort((a, b) => +a.id.slice(1) - +b.id.slice(1));
  if (!list.length) return "";
  return list.map((s) => `<details class="source" id="src-${esc(s.id)}">
      <summary><span class="sid">[${esc(s.id)}]</span><span class="kind">${esc(KIND_LABELS[s.kind] || s.kind)}</span>
        <strong>${esc(s.title)}</strong><span class="loc">${esc(s.location)}</span></summary>
      <div class="loc">${esc(s.publisher)}</div>
      <blockquote>${esc(s.excerpt)}</blockquote>
      ${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.url)}</a>` : ""}
    </details>`).join("");
}

function renderSources() {
  let card = $("#sources-card");
  if (!card) {
    const results = $("#results");
    results.querySelector(".empty-state")?.remove();
    results.insertAdjacentHTML("beforeend", `<div class="card" id="sources-card"><h2>📚 Sources</h2><div class="sources-list"></div></div>`);
    card = $("#sources-card");
  }
  card.querySelector(".sources-list").innerHTML = sourcesHtml();
}

function renderResults(r) {
  const doc = r.document || {};
  const meta = [doc.type, doc.date && `Date: ${doc.date}`, doc.doctor?.name && `Doctor: ${doc.doctor.name}`, `Read via ${r.extraction.method}`]
    .filter(Boolean).map((m) => `<span class="pill">${esc(m)}</span>`).join("");

  const warnings = [
    ...r.extraction.warnings.map((w) => esc(w)),
    ...r.warnings.map((w) => `${esc(w.text)} ${chips(w.sources)}`),
  ];

  $("#results").innerHTML = `
    <div class="card summary-card">
      <h2>📝 In simple words</h2>
      <div class="meta">${meta}</div>
      <p class="summary-text">${esc(r.summary || "See the details below.")}</p>
      <button class="voice-btn" id="hindi-summary-btn">🔊 हिंदी में सुनें (Listen in Hindi)</button>
      <div id="hindi-summary" class="hindi-box hidden" lang="hi"></div>
    </div>
    ${listCard("⚠️ Please note", warnings, "warn-card")}
    ${r.medicines.length ? `<div class="section-title">Medicines (${r.medicines.length})</div>` + r.medicines.map(medicineCard).join("") : ""}
    ${r.conditions.length ? `<div class="section-title">Diagnosis</div>` + r.conditions.map(conditionCard).join("") : ""}
    ${r.tests.length ? `<div class="section-title">Tests advised</div>` + r.tests.map(testCard).join("") : ""}
    ${listCard("👨‍⚕️ Doctor's other advice", [...r.advice, r.follow_up && `Follow-up: ${r.follow_up}`].filter(Boolean).map(esc))}
    ${listCard("❓ Could not read clearly", r.unclear.map(esc), "warn-card")}
    ${listCard("💬 Questions you can ask your doctor", r.questions.map(esc))}
    <div class="card" id="sources-card"><h2>📚 Sources</h2><div class="sources-list">${sourcesHtml()}</div></div>
    <div class="card"><details class="raw"><summary>Text we read from your document (check it matches)</summary>
      <pre>${esc(r.extraction.text)}</pre></details></div>
    <p class="disclaimer">${esc(r.disclaimer)}</p>`;

  $("#hindi-summary-btn").addEventListener("click", (e) =>
    speakHindi({ session_id: state.sessionId, target: "analysis" }, e.currentTarget, $("#hindi-summary")));
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- Hindi voice ----------
function stopAudio() {
  if (state.audio) { state.audio.pause(); state.audio = null; }
  window.speechSynthesis?.cancel();
}

function browserSpeak(text, onEnd) {
  if (!window.speechSynthesis) throw new Error("No Hindi voice available in this browser.");
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "hi-IN";
  const voice = speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith("hi"));
  if (voice) utter.voice = voice;
  utter.onend = onEnd;
  speechSynthesis.speak(utter);
}

async function playHindi(text, onEnd) {
  stopAudio();
  try {
    const res = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (!res.ok) throw new Error("server tts failed");
    const audio = new Audio(URL.createObjectURL(await res.blob()));
    audio.onended = onEnd;
    state.audio = audio;
    await audio.play();
  } catch {
    browserSpeak(text, onEnd); // offline fallback: the browser's own Hindi voice
  }
}

async function speakHindi(payload, btn, box) {
  if (btn.dataset.playing === "1") {
    stopAudio();
    btn.dataset.playing = "0";
    btn.textContent = btn.dataset.label;
    return;
  }
  btn.dataset.label = btn.dataset.label || btn.textContent;
  btn.disabled = true;
  btn.textContent = "⏳ हिंदी तैयार हो रही है…";
  try {
    const { hindi } = await api("/api/hindi", payload);
    if (box) { box.textContent = hindi; box.classList.remove("hidden"); }
    btn.dataset.playing = "1";
    btn.textContent = "⏹ Stop";
    await playHindi(hindi, () => {
      btn.dataset.playing = "0";
      btn.textContent = btn.dataset.label;
    });
  } catch (err) {
    btn.dataset.playing = "0";
    btn.textContent = btn.dataset.label;
    alert(err.message);
  } finally {
    btn.disabled = false;
  }
}

// ---------- chat ----------
function addMessage(role, html, extraClass = "") {
  const div = document.createElement("div");
  div.className = `msg ${role} ${extraClass}`;
  div.innerHTML = html;
  $("#chat-log").appendChild(div);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return div;
}

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  const fromVoice = state.voiceQuestion;
  state.voiceQuestion = false;
  input.value = "";
  addMessage("user", esc(message));
  const pending = addMessage("bot", `<span class="spinner"></span>`);
  try {
    const r = await api("/api/chat", { session_id: state.sessionId, message });
    state.sessionId = r.session_id;
    rememberSources(r.sources);
    if (r.sources.length) renderSources();
    pending.remove();
    if (r.emergency) addMessage("bot", `🚨 ${esc(r.emergency_note)}`, "emergency");
    const msg = addMessage("bot", withInlineCites(r.answer) +
      `<div class="msg-actions"><button class="voice-btn small">🔊 हिंदी</button></div><div class="hindi-box hidden" lang="hi"></div>`);
    const btn = msg.querySelector(".voice-btn");
    const box = msg.querySelector(".hindi-box");
    btn.addEventListener("click", () => speakHindi({ text: r.answer }, btn, box));
    if (fromVoice || $("#auto-hindi").checked) btn.click();
  } catch (err) {
    pending.remove();
    addMessage("bot", `⚠️ ${esc(err.message)}`, "emergency");
  }
});

// Voice input (Web Speech API – Chrome / Edge)
(function setupMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const mic = $("#mic-btn");
  if (!SR) {
    mic.disabled = true;
    mic.title = "Voice input needs Chrome or Edge";
    return;
  }
  let rec = null;
  mic.addEventListener("click", () => {
    if (rec) return rec.stop();
    rec = new SR();
    rec.lang = $("#mic-lang").value;
    rec.interimResults = false;
    rec.onresult = (e) => {
      $("#chat-input").value = e.results[0][0].transcript;
      state.voiceQuestion = true;
      $("#chat-form").requestSubmit();
    };
    rec.onend = () => { rec = null; mic.classList.remove("listening"); };
    rec.onerror = () => { rec = null; mic.classList.remove("listening"); };
    mic.classList.add("listening");
    rec.start();
  });
})();

window.speechSynthesis?.getVoices(); // preload voices list
loadStatus();
