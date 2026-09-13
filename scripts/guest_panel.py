#!/usr/bin/env python3
"""
A no-terminal control panel for a friend lending a machine overnight.

    .venv/bin/python scripts/guest_panel.py

Then open http://localhost:8420 in a browser. Type a name, hit Start, and
watch the status update -- no other command ever needs to be typed or
copy-pasted. Standard library only, same as upload_results.py.

What this actually does under the hood is exactly what a person would type
by hand: it launches run_montecarlo.py --overnight as a detached background
process (the same thing nohup ... & does), and separately shells out to
upload_results.py when asked to send results. This script is just a
friendlier front end for those two commands, not a different code path.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / ".guest_state.json"
LOG_PATH = ROOT / "run.log"
UPLOAD_URL = "https://cuboatrace2026-results.netlify.app/api/upload"
UPLOAD_TOKEN = "PW5hxDurVGmVN1AjT5rDxrKf4nvdndUG"
PYTHON = str(ROOT / ".venv" / "bin" / "python")
PORT = 8420
SHORTCUT_NAME = "Canoe Hull Search.url"

_lock = threading.Lock()


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def open_browser(url: str) -> None:
    """webbrowser.open() can't reach a real browser from inside WSL -- there
    is no browser installed in the Linux side, so it has to be handed off
    to Windows explicitly instead."""
    if is_wsl():
        for cmd in (["cmd.exe", "/c", "start", "", url],
                    ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


def windows_desktop_path() -> Path | None:
    """The friend's real Windows desktop, reached from inside WSL. Returns
    None if it can't be found (e.g. running outside WSL/Windows)."""
    if not is_wsl():
        candidate = Path.home() / "Desktop"
        return candidate if candidate.is_dir() else None
    try:
        win_path = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        if not win_path:
            return None
        linux_path = subprocess.run(
            ["wslpath", "-u", win_path],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        path = Path(linux_path)
        return path if path.is_dir() else None
    except (OSError, subprocess.SubprocessError):
        return None


def create_desktop_shortcut(url: str) -> dict:
    desktop = windows_desktop_path()
    if not desktop:
        return {"ok": False, "error": "Couldn't find your desktop folder automatically."}
    try:
        (desktop / SHORTCUT_NAME).write_text(f"[InternetShortcut]\nURL={url}\n")
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": str(desktop / SHORTCUT_NAME)}


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def is_running(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError):
        return False


def tail_lines(path: Path, n: int = 500) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-n:]


PROGRESS_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+\S+\s+\(([\d.]+)h elapsed, ([\d.]+)h left\)")
ERROR_RE = re.compile(r"(error:|Traceback)")
SUMMARY_RE = re.compile(r"most robust:")


def summarize_log(lines: list[str]) -> dict:
    text = "\n".join(lines)
    if ERROR_RE.search(text):
        detail = lines[-1] if lines else "unknown error"
        return {"phase": "error", "detail": detail}
    if SUMMARY_RE.search(text):
        return {"phase": "done", "detail": "Finished -- ready to send results."}
    match = None
    for line in reversed(lines):
        match = PROGRESS_RE.search(line)
        if match:
            break
    if match:
        i, n, elapsed, left = match.groups()
        pct = round(int(i) / max(int(n), 1) * 100)
        return {
            "phase": "searching",
            "detail": f"Testing hull {i} of {n} -- {elapsed}h elapsed, {left}h left",
            "progress": pct,
        }
    if "stage 1:" in text and "feasible" in text:
        return {"phase": "searching", "detail": "Starting the timed search..."}
    if "stage 1: sampling" in text:
        return {
            "phase": "preparing",
            "detail": ("Preparing -- checking 60,000 possible hull shapes "
                       "before the timed search starts. This can take several "
                       "minutes, longer on a slower computer."),
        }
    if lines:
        return {"phase": "starting", "detail": lines[-1]}
    return {"phase": "idle", "detail": ""}


def current_status() -> dict:
    with _lock:
        state = load_state()
    pid = state.get("pid")
    running = is_running(pid)
    lines = tail_lines(LOG_PATH)
    log_summary = summarize_log(lines) if (running or lines) else {"phase": "idle", "detail": ""}
    phase = log_summary["phase"]
    if not running and phase not in ("done", "error"):
        phase = "stopped" if state.get("started_at") else "idle"
    return {
        "name": state.get("name", ""),
        "hours": state.get("hours"),
        "seed": state.get("seed"),
        "started_at": state.get("started_at"),
        "running": running,
        "phase": phase,
        "detail": log_summary.get("detail", ""),
        "progress": log_summary.get("progress"),
        "log_tail": lines[-40:],
    }


def start_run(name: str, hours: float) -> dict:
    with _lock:
        state = load_state()
        if is_running(state.get("pid")):
            return {"ok": False, "error": "A run is already going -- stop it first."}
        seed = int.from_bytes(os.urandom(2), "big")
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        log_fh = open(LOG_PATH, "w")
        cmd = [
            PYTHON, "scripts/run_montecarlo.py", "--overnight",
            "--hours", str(hours), "--seed", str(seed),
            "--material", "paperboard_unknown",
        ]
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        log_fh.close()
        state = {
            "name": name, "hours": hours, "seed": seed,
            "pid": proc.pid, "started_at": time.time(),
        }
        save_state(state)
        return {"ok": True, "seed": seed, "pid": proc.pid}


def stop_run() -> dict:
    with _lock:
        state = load_state()
    pid = state.get("pid")
    if not is_running(pid):
        return {"ok": False, "error": "Nothing is running."}
    os.kill(pid, signal.SIGINT)
    return {"ok": True}


def run_quick_test() -> dict:
    cmd = [
        PYTHON, "scripts/run_montecarlo.py", "--named", "recommended",
        "-n", "50", "--seed", "1", "--material", "paperboard_unknown",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "Timed out after 60s -- something is wrong with the setup."}
    ok = result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    return {"ok": ok, "output": output.strip()}


def run_upload(name: str) -> dict:
    if not name:
        return {"ok": False, "output": "Type your name first."}
    env = dict(os.environ, UPLOAD_TOKEN=UPLOAD_TOKEN)
    cmd = [PYTHON, "scripts/upload_results.py", "--tag", name, "--url", UPLOAD_URL]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "Upload timed out -- check your internet connection."}
    ok = result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    return {"ok": ok, "output": output.strip()}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Canoe Hull Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#1a2332; --ink-soft:#4a5568; --ground:#f6f3ec; --panel:#ffffff;
  --line:#ddd4bf; --line-soft:#eae3d3;
  --accent:#b8501f; --accent-ink:#8a3c15; --accent-soft:#f1e2d3;
  --mono-bg:#1a2332; --mono-text:#e9e4d6; --mono-dim:#8b93a3;
  --good:#3f7d5c; --good-soft:#e3efe8; --warn:#a6461f; --warn-soft:#f6e3db;
  --font-display:"Archivo",system-ui,sans-serif;
  --font-body:"Public Sans",system-ui,-apple-system,sans-serif;
  --font-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ink:#eae4d6; --ink-soft:#a8a396; --ground:#181d27; --panel:#1f2530;
    --line:#333c4a; --line-soft:#2a3140;
    --accent:#e8a56a; --accent-ink:#f2c193; --accent-soft:#332216;
    --mono-bg:#0f131b; --mono-text:#e9e4d6; --mono-dim:#6b7280;
    --good:#6bbf94; --good-soft:#1c2b23; --warn:#e08a5c; --warn-soft:#332019;
  }
}
:root[data-theme="dark"]{
  --ink:#eae4d6; --ink-soft:#a8a396; --ground:#181d27; --panel:#1f2530;
  --line:#333c4a; --line-soft:#2a3140;
  --accent:#e8a56a; --accent-ink:#f2c193; --accent-soft:#332216;
  --mono-bg:#0f131b; --mono-text:#e9e4d6; --mono-dim:#6b7280;
  --good:#6bbf94; --good-soft:#1c2b23; --warn:#e08a5c; --warn-soft:#332019;
}
*{box-sizing:border-box}
body{margin:0; background:var(--ground); color:var(--ink); font-family:var(--font-body); font-size:16px; line-height:1.55}
.wrap{max-width:560px; margin:0 auto; padding:48px 22px 80px}
h1{font-family:var(--font-display); font-weight:800; font-size:clamp(26px,5vw,34px); margin:0 0 6px; letter-spacing:-0.01em}
.dek{color:var(--ink-soft); margin:0 0 32px; max-width:52ch}
.card{background:var(--panel); border:1px solid var(--line-soft); border-radius:14px; padding:24px; margin-bottom:20px}
label{display:block; font-family:var(--font-display); font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:0.06em; color:var(--accent-ink); margin-bottom:6px}
input[type=text],input[type=number]{
  width:100%; font-family:var(--font-mono); font-size:15px; padding:10px 12px;
  border:1px solid var(--line); border-radius:8px; background:var(--ground); color:var(--ink);
}
input:focus{outline:2px solid var(--accent); outline-offset:1px}
.row{display:flex; gap:14px; flex-wrap:wrap}
.row > div{flex:1; min-width:120px}
.field{margin-bottom:16px}
.hint{font-size:13px; color:var(--ink-soft); margin-top:6px}
button{
  font-family:var(--font-display); font-weight:700; font-size:14px; cursor:pointer;
  border:none; border-radius:9px; padding:12px 18px; transition:opacity 0.15s;
}
button:disabled{opacity:0.45; cursor:default}
.btn-primary{background:var(--accent); color:#fff9f2}
.btn-primary:hover:not(:disabled){opacity:0.9}
.btn-secondary{background:var(--accent-soft); color:var(--accent-ink)}
.btn-secondary:hover:not(:disabled){opacity:0.85}
.btn-danger{background:var(--warn-soft); color:var(--warn)}
.btn-danger:hover:not(:disabled){opacity:0.85}
.actions{display:flex; gap:10px; flex-wrap:wrap; margin-top:6px}
.status{display:flex; align-items:center; gap:10px; margin-bottom:14px}
.dot{width:10px; height:10px; border-radius:50%; background:var(--ink-soft); flex-shrink:0}
.dot.searching,.dot.preparing,.dot.starting{background:var(--accent); animation:pulse 1.4s infinite}
.dot.done{background:var(--good)}
.dot.error{background:var(--warn)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.35}}
.phase-label{font-family:var(--font-display); font-weight:700; font-size:15px}
.detail{color:var(--ink-soft); font-size:14px; margin:0 0 14px}
.bar{height:8px; border-radius:5px; background:var(--line-soft); overflow:hidden; margin-bottom:14px}
.bar-fill{height:100%; background:var(--accent); transition:width 0.4s ease}
.result-box{
  background:var(--mono-bg); color:var(--mono-text); font-family:var(--font-mono); font-size:12.5px;
  border-radius:8px; padding:12px 14px; white-space:pre-wrap; word-break:break-word; max-height:200px; overflow-y:auto; margin-top:10px;
}
.result-box.ok{background:var(--good-soft); color:var(--ink)}
.result-box.bad{background:var(--warn-soft); color:var(--ink)}
details{margin-top:6px}
summary{cursor:pointer; font-size:13px; color:var(--ink-soft)}
.footer{color:var(--ink-soft); font-size:13px; text-align:center; margin-top:36px}
.shortcut-bar{
  display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;
  background:var(--accent-soft); border-radius:10px; padding:12px 16px; margin-bottom:24px;
  font-size:13.5px; color:var(--accent-ink);
}
.shortcut-bar button{white-space:nowrap}
.shortcut-bar .dismiss{background:none; color:var(--accent-ink); text-decoration:underline; padding:6px 4px}
[hidden]{display:none !important}
</style></head><body>
<div class="wrap">
  <h1>Canoe Hull Search</h1>
  <p class="dek">Type your name, hit start, and leave this tab open (or close it -- the search keeps going either way). Check back here any time to see how it's doing.</p>

  <div class="shortcut-bar" id="shortcut-bar">
    <span>Want an icon for this page on your desktop, so you don't need the terminal to get back here?</span>
    <span>
      <button class="btn-secondary" id="btn-shortcut">Add to desktop</button>
      <button class="dismiss" id="btn-shortcut-dismiss">No thanks</button>
    </span>
  </div>
  <div id="shortcut-result" class="result-box" hidden style="margin-bottom:20px"></div>

  <div class="card">
    <div class="field">
      <label for="name">Your name</label>
      <input type="text" id="name" placeholder="e.g. Aiden" autocomplete="off">
    </div>
    <div class="row">
      <div class="field">
        <label for="hours">Hours to run</label>
        <input type="number" id="hours" value="8" min="0.02" step="0.5">
      </div>
    </div>
    <p class="hint">8 hours is a good overnight default. It measures its own speed first, then fits the search into whatever window you give it -- it will never run long.</p>
    <div class="actions">
      <button class="btn-secondary" id="btn-test">Test setup (10 sec)</button>
      <button class="btn-primary" id="btn-start">Start run</button>
    </div>
    <div id="test-result" class="result-box" hidden></div>
  </div>

  <div class="card" id="status-card" hidden>
    <div class="status">
      <span class="dot" id="status-dot"></span>
      <span class="phase-label" id="status-label">--</span>
    </div>
    <div class="bar" id="bar-wrap" hidden><div class="bar-fill" id="bar-fill" style="width:0%"></div></div>
    <p class="detail" id="status-detail"></p>
    <div class="actions">
      <button class="btn-danger" id="btn-stop">Stop</button>
      <button class="btn-primary" id="btn-upload" hidden>Send results to Levi</button>
    </div>
    <div id="upload-result" class="result-box" hidden></div>
    <details>
      <summary>Show raw log</summary>
      <div class="result-box" id="log-box" style="max-height:260px"></div>
    </details>
  </div>

  <p class="footer">Part of a cardboard canoe hull simulator for a college engineering class's lake race.</p>
</div>

<script>
const $ = id => document.getElementById(id);
const nameEl = $('name'), hoursEl = $('hours');

nameEl.value = localStorage.getItem('guestName') || '';
nameEl.addEventListener('input', () => localStorage.setItem('guestName', nameEl.value));

async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}

if (localStorage.getItem('shortcutDismissed')) $('shortcut-bar').hidden = true;

$('btn-shortcut-dismiss').addEventListener('click', () => {
  localStorage.setItem('shortcutDismissed', '1');
  $('shortcut-bar').hidden = true;
});

$('btn-shortcut').addEventListener('click', async () => {
  const btn = $('btn-shortcut');
  btn.disabled = true; btn.textContent = 'Adding...';
  const r = await api('/api/add_shortcut', { method: 'POST' });
  const box = $('shortcut-result');
  box.hidden = false; box.className = 'result-box';
  if (r.ok) {
    box.classList.add('ok');
    box.textContent = 'Done! Look for "Canoe Hull Search" on your desktop -- double-click it any time to get back to this page.';
    localStorage.setItem('shortcutDismissed', '1');
    $('shortcut-bar').hidden = true;
  } else {
    box.classList.add('bad');
    box.textContent = "Couldn't add it automatically: " + r.error;
  }
  btn.disabled = false; btn.textContent = 'Add to desktop';
});

$('btn-test').addEventListener('click', async () => {
  const btn = $('btn-test');
  btn.disabled = true; btn.textContent = 'Testing...';
  const box = $('test-result');
  box.hidden = false; box.className = 'result-box'; box.textContent = 'Running a 10-second check...';
  const r = await api('/api/test', { method: 'POST' });
  box.classList.add(r.ok ? 'ok' : 'bad');
  box.textContent = r.ok ? 'Setup looks good! Safe to start a real run.\\n\\n' + r.output : 'Something is wrong:\\n\\n' + r.output;
  btn.disabled = false; btn.textContent = 'Test setup (10 sec)';
});

$('btn-start').addEventListener('click', async () => {
  const name = nameEl.value.trim();
  if (!name) { nameEl.focus(); return; }
  const hours = parseFloat(hoursEl.value) || 8;
  const btn = $('btn-start');
  btn.disabled = true; btn.textContent = 'Starting...';
  const r = await api('/api/start', {
    method: 'POST', headers: {'content-type':'application/json'},
    body: JSON.stringify({ name, hours }),
  });
  btn.disabled = false; btn.textContent = 'Start run';
  if (!r.ok) { alert(r.error); return; }
  poll();
});

$('btn-stop').addEventListener('click', async () => {
  if (!confirm('Stop the run? It finishes whatever hull it is checking first, then saves progress cleanly.')) return;
  await api('/api/stop', { method: 'POST' });
  poll();
});

$('btn-upload').addEventListener('click', async () => {
  const btn = $('btn-upload');
  btn.disabled = true; btn.textContent = 'Sending...';
  const box = $('upload-result');
  box.hidden = false; box.className = 'result-box'; box.textContent = 'Uploading...';
  const r = await api('/api/upload', {
    method: 'POST', headers: {'content-type':'application/json'},
    body: JSON.stringify({ name: nameEl.value.trim() }),
  });
  box.classList.add(r.ok ? 'ok' : 'bad');
  box.textContent = r.ok ? 'Sent! Thank you for the help.\\n\\n' + r.output : 'Upload failed:\\n\\n' + r.output;
  btn.disabled = false; btn.textContent = 'Send results to Levi';
});

const PHASE_LABEL = {
  idle: 'Not started', starting: 'Starting...', preparing: 'Preparing...',
  searching: 'Searching for the best hull', done: 'Finished',
  stopped: 'Stopped', error: 'Something went wrong',
};

let timer = null;
async function poll() {
  const s = await api('/api/status');
  if (s.name) nameEl.value = s.name;
  if (s.hours) hoursEl.value = s.hours;
  const card = $('status-card');
  card.hidden = s.phase === 'idle' && !s.started_at;
  $('status-dot').className = 'dot ' + s.phase;
  $('status-label').textContent = PHASE_LABEL[s.phase] || s.phase;
  $('status-detail').textContent = s.detail || '';
  $('bar-wrap').hidden = s.progress == null;
  if (s.progress != null) $('bar-fill').style.width = s.progress + '%';
  $('btn-stop').hidden = !s.running;
  $('btn-upload').hidden = !(s.phase === 'done' || s.phase === 'stopped');
  $('log-box').textContent = (s.log_tail || []).join('\\n') || '(nothing yet)';
  $('btn-start').disabled = s.running;
  clearTimeout(timer);
  timer = setTimeout(poll, s.running ? 3000 : 8000);
}
poll();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._send_json(current_status())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/start":
            data = self._read_json()
            name = str(data.get("name", "")).strip()[:40] or "guest"
            try:
                hours = max(0.02, min(48.0, float(data.get("hours", 8))))
            except (TypeError, ValueError):
                hours = 8.0
            self._send_json(start_run(name, hours))
        elif self.path == "/api/stop":
            self._send_json(stop_run())
        elif self.path == "/api/test":
            self._send_json(run_quick_test())
        elif self.path == "/api/upload":
            data = self._read_json()
            name = str(data.get("name", "")).strip()[:40]
            self._send_json(run_upload(name))
        elif self.path == "/api/add_shortcut":
            self._send_json(create_desktop_shortcut(f"http://localhost:{PORT}"))
        else:
            self.send_response(404)
            self.end_headers()


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Panel running -- open {url} in your browser.")
    open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
