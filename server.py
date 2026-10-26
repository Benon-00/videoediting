"""
server.py - tiny local bridge between the Edit Studio frontend and your scripts.

Run it from anywhere:
    python server.py --project "C:\\path\\to\\your\\project"
then open http://127.0.0.1:8765

It uses ONLY the Python standard library (no pip install needed).

What it does
  GET  /api/scripts   -> which scripts it found in the project folder
  GET  /api/config    -> reads the CONFIG block of a script (via ast, never executes it)
  POST /api/config    -> writes changed values back into the .py file (a .bak copy is made once)
  POST /api/run       -> runs a script (cwd = project folder) and captures its output
  GET  /api/job       -> live log of the running job
  POST /api/stop      -> stops the running job
"""
import argparse, ast, json, os, re, shutil, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.getcwd()
PORT = 8765

# id -> candidate filenames (first one that exists in the project folder wins)
SCRIPTS = {
    "transcribe":   ["transcribe.py"],
    "clip_video":   ["clip_video.py"],
    "clean_audio":  ["clean_audio.py"],
    "analyzer":     ["edit_analyzer_full.py"],
    "beat_sync":    ["beat_sync_editor_new.py", "beat_sync_editor.py"],
}

START_RE = re.compile(r"^# =+ CONFIG =+\s*$")
END_RE = re.compile(r"^# =+\s*$")
GROUP_RE = re.compile(r"^#\s*---\s*(.*?)\s*(?:---)?\s*$")

JOB = {"proc": None, "lines": [], "running": False, "rc": None, "script": None, "lock": threading.Lock()}


def script_path(sid):
    for name in SCRIPTS.get(sid, []):
        p = os.path.join(PROJECT, name)
        if os.path.isfile(p):
            return p
    return None


def kind_of(v):
    if isinstance(v, bool): return "bool"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str): return "str"
    if v is None: return "none"
    return "expr"  # list / dict / tuple / set


def read_config(path):
    text = open(path, encoding="utf-8").read()
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if START_RE.match(l)), None)
    if start is None:
        raise ValueError("No '# ===== CONFIG =====' block found in this script.")
    end = next((i for i in range(start + 1, len(lines)) if END_RE.match(lines[i])), len(lines))
    tree = ast.parse(text)
    items, group = [], "General"
    assigns = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and start < node.lineno - 1 < end):
            assigns[node.lineno - 1] = node
    for i in range(start + 1, end):
        m = GROUP_RE.match(lines[i])
        if m and m.group(1):
            group = m.group(1)
        node = assigns.get(i)
        if not node:
            continue
        name = node.targets[0].id
        try:
            val = ast.literal_eval(node.value)
        except Exception:
            continue  # derived/computed value - not editable here
        src = ast.get_source_segment(text, node.value)
        items.append({"name": name, "group": group, "kind": kind_of(val), "value": src,
                      "val": val if kind_of(val) in ("bool", "int", "float", "str") else None,
                      "line": node.lineno})
    return items


def write_config(path, changes):
    errors = {}
    for name, expr in changes.items():
        try:
            ast.literal_eval(expr)
        except Exception as e:
            errors[name] = f"Not a valid Python value: {e}"
    if errors:
        return errors
    text = open(path, encoding="utf-8").read()
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    lines = text.split("\n")
    tree = ast.parse(text)
    start = next(i for i, l in enumerate(lines) if START_RE.match(l))
    todo = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in changes and node.lineno - 1 > start):
            todo.append(node)
    for node in sorted(todo, key=lambda n: -n.lineno):
        name = node.targets[0].id
        a, b = node.lineno - 1, node.end_lineno - 1
        tail = ""
        if a == b:  # keep a trailing "# comment" on one-line assignments
            tail = lines[b][node.value.end_col_offset:]
        new = f"{name} = {changes[name]}{tail}"
        lines[a:b + 1] = new.split("\n")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    return {}


def reader(proc):
    buf, transient = "", False
    def commit(s, tr):
        nonlocal transient
        with JOB["lock"]:
            if transient and JOB["lines"]:
                JOB["lines"][-1] = s
            else:
                JOB["lines"].append(s)
        transient = tr
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        c = ch.decode("utf-8", "replace")
        if c == "\n":
            commit(buf, False); buf = ""
        elif c == "\r":
            if buf: commit(buf, True)
            buf = ""
        else:
            buf += c
    if buf: commit(buf, False)
    proc.wait()
    with JOB["lock"]:
        JOB["running"], JOB["rc"] = False, proc.returncode
        JOB["lines"].append(f"--- finished (exit code {proc.returncode}) ---")


def start_job(sid, args):
    p = script_path(sid)
    if not p:
        return {"error": f"Script for '{sid}' not found in {PROJECT}"}, 404
    with JOB["lock"]:
        if JOB["running"]:
            return {"error": "Another job is already running."}, 409
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "-u", p] + [str(a) for a in args]
        proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        JOB.update(proc=proc, lines=[f"$ {' '.join(cmd[1:])}"], running=True, rc=None, script=sid)
    threading.Thread(target=reader, args=(proc,), daemon=True).start()
    return {"ok": True}, 200


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _origin_ok(self):
        o = self.headers.get("Origin")
        return o is None or o in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}")

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if not u.path.startswith("/api/"):
            name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
            fp = os.path.normpath(os.path.join(HERE, name))
            if not fp.startswith(HERE) or not os.path.isfile(fp):
                self.send_response(404); self.end_headers(); return
            ct = {"html": "text/html", "css": "text/css", "js": "application/javascript"}.get(fp.rsplit(".", 1)[-1], "text/plain")
            data = open(fp, "rb").read()
            self.send_response(200); self.send_header("Content-Type", ct + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        if not self._origin_ok(): return self._json({"error": "bad origin"}, 403)
        if u.path == "/api/scripts":
            out = {}
            for sid in SCRIPTS:
                p = script_path(sid)
                out[sid] = {"found": bool(p), "file": os.path.basename(p) if p else SCRIPTS[sid][0]}
            extra = {n: os.path.isfile(os.path.join(PROJECT, n)) for n in ("webrtcvad.py", "yolov8n.pt")}
            return self._json({"project": PROJECT, "scripts": out, "extras": extra})
        if u.path == "/api/config":
            p = script_path(q.get("script", [""])[0])
            if not p: return self._json({"error": "script not found in project folder"}, 404)
            try: return self._json({"items": read_config(p)})
            except Exception as e: return self._json({"error": str(e)}, 400)
        if u.path == "/api/job":
            since = int(q.get("since", ["0"])[0])
            with JOB["lock"]:
                lines = JOB["lines"]
                # a replaced "transient" last line is re-sent by always returning from since-1
                return self._json({"running": JOB["running"], "rc": JOB["rc"], "script": JOB["script"],
                                   "lines": lines[max(0, since - 1):], "from": max(0, since - 1), "next": len(lines)})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._origin_ok() or self.headers.get("X-Studio") != "1":
            return self._json({"error": "forbidden"}, 403)
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        u = urlparse(self.path)
        if u.path == "/api/config":
            p = script_path(body.get("script", ""))
            if not p: return self._json({"error": "script not found"}, 404)
            errs = write_config(p, body.get("changes", {}))
            return self._json({"errors": errs}, 400 if errs else 200)
        if u.path == "/api/run":
            res, code = start_job(body.get("script", ""), body.get("args", []))
            return self._json(res, code)
        if u.path == "/api/stop":
            with JOB["lock"]:
                pr = JOB["proc"]
            if pr and JOB["running"]:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pr.pid)], capture_output=True)
                else:
                    pr.terminate()
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.getcwd(), help="folder that contains your scripts + data folders")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    PROJECT, PORT = os.path.abspath(a.project), a.port
    print(f"Edit Studio -> http://127.0.0.1:{PORT}\nProject folder: {PROJECT}\nCtrl+C to quit.")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
