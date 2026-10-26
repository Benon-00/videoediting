const $ = (s, r = document) => r.querySelector(s);
const el = (t, a = {}, ...kids) => {
  const e = document.createElement(t);
  for (const [k, v] of Object.entries(a)) {
    if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  kids.flat().forEach(c => e.append(c?.nodeType ? c : document.createTextNode(c ?? "")));
  return e;
};
const api = async (path, body) => {
  const r = await fetch(path, body ? {method: "POST", headers: {"Content-Type": "application/json", "X-Studio": "1"}, body: JSON.stringify(body)} : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok && !j.errors) throw new Error(j.error || r.statusText);
  return j;
};
const toast = m => { const t = $(".toast") || document.body.appendChild(el("div", {class: "toast"})); t.textContent = m; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200); };

const ORDER = ["transcribe", "clip_video", "clean_audio", "analyzer", "beat_sync"];
let STATE = {info: null, tab: "overview", cfg: {}, dirty: {}, poll: null, since: 0};

async function init() {
  try {
    STATE.info = await api("/api/scripts");
    $("#conn").textContent = "Connected to local scripts"; $("#conn").className = "conn ok";
    $("#proj").textContent = "Project: " + STATE.info.project;
  } catch (e) {
    STATE.info = null;
    $("#conn").textContent = "Server not running"; $("#conn").className = "conn bad";
  }
  renderNav(); show(STATE.tab);
}

function renderNav() {
  const nav = $("#nav"); nav.innerHTML = "";
  const add = (id, label, n, found) => nav.append(el("button", {class: STATE.tab === id ? "on" : "", onclick: () => show(id)},
    el("span", {class: "n"}, n), label, found === undefined ? "" : el("span", {class: "dot" + (found ? " ok" : ""), title: found ? "found" : "not found in project folder"})));
  nav.append(el("div", {class: "sep"}, "Guide"));
  add("overview", "How it fits together", "i");
  add("setup", "Connect to your PC", "?");
  nav.append(el("div", {class: "sep"}, "Scripts (pipeline order)"));
  ORDER.forEach(id => add(id, SCRIPT_INFO[id].title.replace(".py", ""), SCRIPT_INFO[id].step, STATE.info?.scripts[id]?.found));
}

function show(tab) {
  if (STATE.dirty[STATE.tab] && Object.keys(STATE.dirty[STATE.tab]).length && tab !== STATE.tab && !confirm("You have unsaved changes in this script. Leave anyway?")) return;
  STATE.tab = tab; renderNav();
  const m = $("#main"); m.innerHTML = "";
  if (tab === "overview") return overview(m);
  if (tab === "setup") return setup(m);
  scriptView(m, tab);
}

function overview(m) {
  m.append(el("h1", {}, "How the scripts fit together"),
    el("p", {class: "lead"}, "Run them left to right. Each script's output folder is the next script's input folder."));
  const flow = el("div", {class: "flow"});
  ORDER.forEach(id => {
    const i = SCRIPT_INFO[id];
    flow.append(el("div", {class: "card step"}, el("div", {class: "num"}, i.step),
      el("div", {}, el("h3", {}, i.title + " — " + i.short), el("p", {}, i.what),
        el("div", {class: "io"}, el("b", {}, "Reads"), i.reads, el("b", {}, "Writes"), i.writes, el("b", {}, "Feeds"), i.feeds))));
  });
  const w = SCRIPT_INFO.webrtcvad;
  flow.append(el("div", {class: "card step"}, el("div", {class: "num"}, "+"),
    el("div", {}, el("h3", {}, "webrtcvad.py + yolov8n.pt — helper files (not run)"),
      el("p", {}, "webrtcvad.py is a stub that lets transcribe.py's speaker detection import cleanly; yolov8n.pt is the YOLO object-detection weights file edit_analyzer_full.py loads. Keep both in the project folder."))));
  m.append(flow,
    el("h2", {}, "Typical data flow"),
    el("pre", {}, `movie.mp4 ──transcribe──▶ movie_transcript.txt
                                  │
remix/Quotes.txt + movie.mp4 ─────┴─clip_video──▶ video_cutaways/Quotes/001.mp4…
                                                      │
                                                      └─clean_audio──▶ cleaned_clips/…

videos_to_analyze/<niche>/*.mp4 ──edit_analyzer_full──▶ dna_modules/<niche>/*.json
                                                              │
source_footage/ + music/song.mp3 + remix/song.txt ────────────┴─beat_sync_editor──▶ outputs/song.mp4
(+ movie transcript via TRANSCRIPT_MAP)`),
    el("div", {class: "note"}, "Two different script formats share the remix/ folder: clip_video.py expects quote lines (optionally ending in [mm:ss–mm:ss]), while beat_sync_editor expects Hook / Rising Action / Conflict / Comeback / Payoff sections, named after the music file."));
}

function setup(m) {
  const found = STATE.info ? Object.values(STATE.info.scripts).filter(s => s.found).length : 0;
  m.append(el("h1", {}, "Connect this page to the scripts on your PC"),
    el("p", {class: "lead"}, "A web page can't touch your files by itself, so a tiny helper (server.py) runs on your PC. It reads/edits the CONFIG block in each script and runs them for you."));
  m.append(el("div", {class: "card"}, el("ol", {class: "steps"},
    el("li", {html: "<b>Put everything in one project folder</b> — the 5 scripts, <code>webrtcvad.py</code>, <code>yolov8n.pt</code>, plus your data folders (<code>remix/</code>, <code>music/</code>, <code>source_footage/</code>, <code>videos_to_analyze/</code> …). The scripts use relative paths, so they must run from this folder."}),
    el("li", {html: "<b>Copy the <code>edit_studio</code> folder</b> (server.py, index.html, style.css, app.js, descriptions.js) anywhere, e.g. next to the project."}),
    el("li", {html: "<b>Check Python</b> (3.9+) in a terminal: <code>python --version</code>. server.py needs no extra packages. Your scripts still need their own pip installs and <code>ffmpeg</code> on PATH."}),
    el("li", {html: "<b>Start the helper</b> from inside the <code>edit_studio</code> folder, pointing at your project:<pre>python server.py --project \"C:\\Users\\you\\MyProject\"</pre>Use the same Python/venv you installed the script packages into."}),
    el("li", {html: "<b>Open <code>http://127.0.0.1:8765</code></b> in your browser (not by double-clicking index.html — the helper serves it). The top-left badge turns green and each script gets a green dot."}),
    el("li", {html: "<b>Edit → Save → Run.</b> Save writes values into the .py file (a one-time <code>.bak</code> backup is made). Run starts the script and streams its output here."}),
    el("li", {html: "<b>Different port?</b> <code>--port 9000</code>, then open that address instead."}))));
  m.append(el("h2", {}, "Status"), el("div", {class: "card"},
    STATE.info ? el("div", {}, el("p", {}, `Project: ${STATE.info.project}`),
      ...ORDER.map(id => el("div", {}, el("span", {class: "pill " + (STATE.info.scripts[id].found ? "ok" : "bad")}, STATE.info.scripts[id].found ? "found" : "missing"), " " + STATE.info.scripts[id].file)),
      ...Object.entries(STATE.info.extras).map(([n, ok]) => el("div", {}, el("span", {class: "pill " + (ok ? "ok" : "bad")}, ok ? "found" : "missing"), " " + n + (n === "yolov8n.pt" && !ok ? " (auto-downloads on first analyzer run if online)" : ""))))
    : el("p", {}, "Server not reachable. Start server.py (step 4), then reload this page.")),
    el("button", {class: "btn", onclick: init}, "Re-check"));
  m.append(el("h2", {}, "Good to know"), el("div", {class: "card"}, el("ul", {},
    el("li", {}, "The helper only listens on 127.0.0.1 (your PC), only runs these five scripts, and rejects requests from other websites."),
    el("li", {}, "One job runs at a time. Closing the browser tab doesn't stop a running job; use Stop."),
    el("li", {}, "Only plain values (numbers, text, True/False, None, lists, dicts) are editable. Computed lines (e.g. TARGET_WIDTH/HEIGHT) follow automatically from what they derive from."),
    el("li", {}, "Edit lists/dicts as Python, e.g. [\"a\", \"b\"] or {\"hook\": 0.35}. Invalid values are rejected before anything is written."),
    el("li", {}, "To undo everything, copy the .bak file over the script."))));
}

const toExpr = (kind, v) => kind === "bool" ? (v ? "True" : "False") : kind === "str" ? JSON.stringify(v) : String(v);

async function scriptView(m, id) {
  const i = SCRIPT_INFO[id];
  m.append(el("h1", {}, i.title), el("p", {class: "lead"}, i.what));
  if (!STATE.info?.scripts[id]?.found) { m.append(el("div", {class: "note"}, "Script not found in the project folder (or server not running). See “Connect to your PC”.")); return; }
  let items;
  try { items = (await api("/api/config?script=" + id)).items; } catch (e) { m.append(el("div", {class: "note"}, e.message)); return; }
  STATE.cfg[id] = items; STATE.dirty[id] = {};
  const desc = CFG_DESC[id] || {};
  const fields = {};

  const search = el("input", {type: "search", placeholder: "Filter settings…", oninput: () => {
    const q = search.value.toLowerCase();
    items.forEach(it => { const f = fields[it.name]; f.row.style.display = (it.name + " " + (desc[it.name] || "")).toLowerCase().includes(q) ? "" : "none"; });
    document.querySelectorAll("details.grp").forEach(d => { d.style.display = [...d.querySelectorAll(".field")].some(f => f.style.display !== "none") ? "" : "none"; if (q) d.open = true; });
  }});
  const saveBtn = el("button", {class: "btn pri", disabled: "", onclick: save}, "Save changes");
  const resetBtn = el("button", {class: "btn", onclick: () => show(id)}, "Discard");
  m.append(el("div", {class: "bar"}, search, saveBtn, resetBtn));

  const groups = {};
  items.forEach(it => (groups[it.group] ||= []).push(it));
  Object.entries(groups).forEach(([g, list], gi) => {
    const d = el("details", {class: "grp"}, el("summary", {}, g, el("span", {class: "pill"}, list.length + "")));
    if (gi === 0) d.open = true;
    list.forEach(it => {
      let input, get, init0;
      if (it.kind === "bool") { input = el("input", {type: "checkbox"}); input.checked = it.val; get = () => input.checked; }
      else if (it.kind === "int" || it.kind === "float") { input = el("input", {type: "number", step: "any", value: it.val}); get = () => input.value === "" ? it.val : Number(input.value); }
      else if (it.kind === "str") { input = el("input", {type: "text", value: it.val}); get = () => input.value; }
      else { const long = it.value.length > 60 || it.value.includes("\n"); input = el(long ? "textarea" : "input", {type: "text"}); input.value = it.value; get = () => input.value; }
      init0 = JSON.stringify(get());
      const err = el("div", {class: "err"});
      const row = el("div", {class: "field"}, el("div", {}, el("div", {class: "nm"}, it.name), el("div", {class: "ds"}, desc[it.name] || "(no description yet)")), el("div", {}, input, err));
      const on = () => {
        const changed = JSON.stringify(get()) !== init0;
        row.classList.toggle("dirty", changed);
        if (changed) STATE.dirty[id][it.name] = toExpr(it.kind, get()).toString(); else delete STATE.dirty[id][it.name];
        if (it.kind === "expr" || it.kind === "none") { if (changed) STATE.dirty[id][it.name] = input.value; }
        saveBtn.disabled = !Object.keys(STATE.dirty[id]).length;
      };
      input.addEventListener("input", on); input.addEventListener("change", on);
      fields[it.name] = {row, err};
      d.append(row);
    });
    m.append(d);
  });

  async function save() {
    Object.values(fields).forEach(f => f.err.textContent = "");
    const r = await api("/api/config", {script: id, changes: STATE.dirty[id]});
    if (r.errors && Object.keys(r.errors).length) {
      Object.entries(r.errors).forEach(([n, msg]) => fields[n] && (fields[n].err.textContent = msg));
      toast("Fix the highlighted values — nothing was saved."); return;
    }
    toast("Saved to " + STATE.info.scripts[id].file); STATE.dirty[id] = {}; show(id);
  }

  // ---- Run panel ----
  m.append(el("h2", {}, "Run"));
  const args = {};
  const box = el("div", {class: "card"});
  if (i.runArgs.length) {
    const g = el("div", {class: "args"});
    i.runArgs.forEach(k => {
      const a = RUN_ARGS[k];
      const inp = a.type === "select" ? el("select", {}, a.opts.map(o => el("option", {value: o}, o || "(use config)"))) : el("input", {type: a.type, step: "any", placeholder: "(use config)"});
      args[k] = inp; g.append(el("div", {}, el("label", {}, a.label), inp, el("div", {class: "ds", style: "font-size:11.5px;color:var(--mute)"}, a.help)));
    });
    box.append(g);
  }
  const log = el("div", {class: "log"}, "Output appears here.");
  const runBtn = el("button", {class: "btn pri", onclick: run}, "▶ Run " + i.title);
  const stopBtn = el("button", {class: "btn danger", onclick: () => api("/api/stop", {}), disabled: ""}, "■ Stop");
  box.append(el("div", {class: "bar"}, runBtn, stopBtn, el("span", {class: "pill", id: "jobstate"}, "idle")), log);
  m.append(box);

  async function run() {
    if (Object.keys(STATE.dirty[id]).length && !confirm("You have unsaved changes — the script will run with the OLD saved values. Run anyway?")) return;
    const a = [];
    i.runArgs.forEach(k => { const v = args[k].value; if (v !== "") a.push(RUN_ARGS[k].flag, v); });
    const r = await api("/api/run", {script: id, args: a}).catch(e => ({error: e.message}));
    if (r.error) return toast(r.error);
    STATE.since = 0; log.textContent = ""; startPoll(log, runBtn, stopBtn);
  }
  api("/api/job?since=0").then(j => { if (j.running && j.script === id) { STATE.since = 0; startPoll(log, runBtn, stopBtn); } });
}

function startPoll(log, runBtn, stopBtn) {
  clearInterval(STATE.poll); runBtn.disabled = true; stopBtn.disabled = false;
  const lines = [];
  const tick = async () => {
    let j; try { j = await api("/api/job?since=" + STATE.since); } catch { return; }
    lines.length = j.from; lines.push(...j.lines); STATE.since = j.next;
    const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
    log.textContent = lines.join("\n"); if (atEnd) log.scrollTop = log.scrollHeight;
    const st = $("#jobstate"); if (st) st.textContent = j.running ? "running…" : (j.rc === 0 ? "finished OK" : "stopped / failed (code " + j.rc + ")");
    if (!j.running) { clearInterval(STATE.poll); runBtn.disabled = false; stopBtn.disabled = true; }
  };
  STATE.poll = setInterval(tick, 800); tick();
}

init();
