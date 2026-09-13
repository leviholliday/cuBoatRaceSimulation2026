#!/usr/bin/env python3
"""
A no-terminal control panel for a friend lending a machine overnight.

    .venv/bin/python scripts/guest_panel.py start      # run in the background, open the browser
    .venv/bin/python scripts/guest_panel.py            # run in this window instead (Windows/WSL)
    .venv/bin/python scripts/guest_panel.py uninstall  # stop everything, remove the icon and this folder

Everything else happens on the page at http://localhost:8420: type a name,
hit Start, watch the status, send the results. Standard library only, same
as upload_results.py.

Under the hood this is exactly what a person would type by hand: it
launches run_montecarlo.py --overnight as a detached background process,
and shells out to upload_results.py when asked to send results.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUEST_INSTALL = Path.home() / "cuBoatRaceSimulation2026"
STATE_PATH = ROOT / ".guest_state.json"
PANEL_PID_PATH = ROOT / ".guest_panel.pid"
LOG_PATH = ROOT / "run.log"
PANEL_LOG_PATH = ROOT / "panel.log"
UPLOAD_URL = "https://cuboat.netlify.app/api/upload"
SEED_URL = "https://cuboat.netlify.app/api/seed"
UPLOAD_TOKEN = "PW5hxDurVGmVN1AjT5rDxrKf4nvdndUG"
PYTHON = str(ROOT / ".venv" / "bin" / "python")
# GUEST_PANEL_PORT overrides this for a machine where 8420 is already taken.
PORT = int(os.environ.get("GUEST_PANEL_PORT", "8420"))
SHORTCUT_BASENAME = "Canoe Hull Search"
SHORTCUT_EXTS = (".bat", ".url", ".command", ".webloc", ".desktop")
# Lets a re-run of the setup command (which pulls updates) tell that the
# panel already running is an older copy, and replace it.
VERSION = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]

_lock = threading.Lock()
_children: list[subprocess.Popen] = []


def reap_children() -> None:
    """The panel launched these, so it has to collect them once they exit --
    otherwise a finished run lingers as a zombie that still looks alive."""
    for child in list(_children):
        if child.poll() is not None:
            try:
                _children.remove(child)
            except ValueError:
                pass


# ---------------------------------------------------------------- environment

def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def host_kind() -> str:
    """'windows' (incl. WSL), 'mac', or 'linux' -- the shortcut file format
    and browser-launch method differ on each."""
    if is_wsl() or platform.system() == "Windows":
        return "windows"
    if platform.system() == "Darwin":
        return "mac"
    return "linux"


def can_open_browser() -> bool:
    """False when there's no screen here to open a browser on: a Linux box
    with no desktop session (a Pi run headless), or any machine reached over
    SSH, where a browser would open on the far end if anywhere."""
    if is_wsl():
        return True
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False
    if host_kind() == "linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def bind_host() -> str:
    # Loopback only, unless the page has to be reached from another device
    # (no screen here), or from Windows through WSL's own network.
    return "0.0.0.0" if is_wsl() or not can_open_browser() else "127.0.0.1"


def lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def open_browser(url: str) -> None:
    """webbrowser.open() can't reach a real browser from inside WSL, and is
    unreliable on minimal Linux desktops (including Raspberry Pi OS)."""
    if not can_open_browser():
        return
    if is_wsl():
        for cmd in (["cmd.exe", "/c", "start", "", url],
                    ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue
        return
    if host_kind() == "linux" and shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ------------------------------------------------------------ desktop shortcut

def desktop_path() -> Path | None:
    """The real desktop folder -- on WSL that's the Windows desktop, reached
    through PowerShell since WSL's own filesystem has no such folder."""
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
            ["wslpath", "-u", win_path], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        path = Path(linux_path)
        return path if path.is_dir() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _desktop_entry_arg(s: str) -> str:
    # Desktop Entry quoting: inside double quotes, " ` $ \ take a backslash,
    # and the file's own string escaping then doubles every backslash.
    if not re.search(r"[\s\"'\\`$<>~|&;*?#()]", s):
        return s
    escaped = (s.replace("\\", "\\\\\\\\").replace('"', '\\\\"')
                .replace("`", "\\\\`").replace("$", "\\\\$"))
    return f'"{escaped}"'


def remove_desktop_shortcuts(desktop: Path, keep: str | None = None) -> None:
    for ext in SHORTCUT_EXTS:
        if ext == keep:
            continue
        try:
            (desktop / f"{SHORTCUT_BASENAME}{ext}").unlink()
        except OSError:
            pass


def create_desktop_shortcut() -> dict:
    """An icon that *starts* the panel and opens it, rather than a plain link
    to localhost -- after a restart the panel isn't running, and a link would
    just open a dead page."""
    desktop = desktop_path()
    if not desktop:
        return {"ok": False, "error": "Couldn't find your desktop folder automatically."}
    script = str(Path(__file__).resolve())
    kind = host_kind()
    try:
        if kind == "windows":
            ext = ".bat"
            distro = os.environ.get("WSL_DISTRO_NAME")
            distro_arg = f"-d {distro} " if distro else ""
            root = str(ROOT).replace("%", "%%")
            content = (
                "@echo off\ntitle Canoe Hull Search\n"
                f'wsl.exe {distro_arg}--cd "{root}" -- .venv/bin/python scripts/guest_panel.py\n'
                "echo.\npause\n"
            )
            path = desktop / f"{SHORTCUT_BASENAME}{ext}"
            path.write_bytes(content.replace("\n", "\r\n").encode())
            note = ("It opens a small Ubuntu window that runs the panel -- leave that "
                    "window open (minimizing is fine) while a search runs.")
        elif kind == "mac":
            ext = ".command"
            path = desktop / f"{SHORTCUT_BASENAME}{ext}"
            path.write_text(f"#!/bin/bash\nexec {shlex.quote(PYTHON)} {shlex.quote(script)} start\n")
            path.chmod(0o755)
            note = "It flashes open a Terminal window to start the panel, then opens your browser."
        else:
            ext = ".desktop"
            path = desktop / f"{SHORTCUT_BASENAME}{ext}"
            path.write_text(
                "[Desktop Entry]\nVersion=1.0\nType=Application\n"
                f"Name={SHORTCUT_BASENAME}\n"
                "Comment=Start and open the hull search control panel\n"
                f"Exec={_desktop_entry_arg(PYTHON)} {_desktop_entry_arg(script)} start\n"
                f"Path={ROOT}\nIcon=applications-internet\nTerminal=false\n"
            )
            path.chmod(0o755)
            note = ('Linux may ask you to right-click it and choose "Allow Launching" '
                    "the first time -- that's a normal one-time security prompt, not an error.")
        remove_desktop_shortcuts(desktop, keep=ext)
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": str(path), "note": note}


# ------------------------------------------------------------------ the search

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def is_running(pid) -> bool:
    """Alive *and* actually the search. Process IDs get reused: a run that
    died hours ago can leave its ID to some unrelated program, which must
    never be shown as the run -- let alone paused or stopped along with it."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, TypeError, OverflowError):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                 capture_output=True, text=True).stdout
    return "run_montecarlo.py" in command


def tail_lines(path: Path, n: int = 500) -> list[str]:
    # The page polls every few seconds and an overnight log keeps growing,
    # so read only its end rather than the whole file each time.
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 256_000))
            data = f.read()
    except OSError:
        return []
    return data.decode(errors="replace").splitlines()[-n:]


PROGRESS_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+\S+\s+\(([\d.]+)h elapsed, ([\d.]+)h left\)")
ERROR_RE = re.compile(r"(error:|Traceback)")
SUMMARY_RE = re.compile(r"most robust:")


def summarize_log(lines: list[str]) -> dict:
    text = "\n".join(lines)
    if ERROR_RE.search(text):
        return {"phase": "error", "detail": lines[-1] if lines else "unknown error"}
    if SUMMARY_RE.search(text):
        return {"phase": "done", "detail": "Finished -- ready to send results."}
    for line in reversed(lines):
        match = PROGRESS_RE.search(line)
        if match:
            i, n, elapsed, left = match.groups()
            return {
                "phase": "searching",
                "detail": f"Testing hull {i} of {n} -- {elapsed}h elapsed, {left}h left",
                "progress": round(int(i) / max(int(n), 1) * 100),
            }
    if "stage 1:" in text and "feasible" in text:
        return {"phase": "searching", "detail": "Starting the timed search..."}
    if "stage 1: sampling" in text:
        return {
            "phase": "preparing",
            "detail": ("Preparing -- checking 60,000 possible hull shapes before the "
                       "timed search starts. This can take several minutes, longer on "
                       "a slower computer."),
        }
    if lines:
        return {"phase": "starting", "detail": lines[-1]}
    return {"phase": "idle", "detail": ""}


def current_status() -> dict:
    with _lock:
        state = load_state()
    reap_children()
    running = is_running(state.get("pid"))
    lines = tail_lines(LOG_PATH)
    summary = summarize_log(lines) if (running or lines) else {"phase": "idle", "detail": ""}
    phase = summary["phase"]
    if not running and phase not in ("done", "error"):
        phase = "stopped" if state.get("started_at") else "idle"
    return {
        "version": VERSION,
        "wsl": is_wsl(),
        "name": state.get("name", ""),
        "hours": state.get("hours"),
        "seed": state.get("seed"),
        "started_at": state.get("started_at"),
        "running": running,
        "phase": phase,
        "detail": summary.get("detail", ""),
        "progress": summary.get("progress"),
        "share": bool(state.get("share")),
        "sharing": _governor.mode if (_governor and running) else None,
        "log_tail": lines[-40:],
    }


def claim_seed(name: str) -> int | None:
    """The next unused seed from the results site, so no two computers repeat
    each other's work. None if the site can't be reached."""
    req = urllib.request.Request(
        SEED_URL, data=json.dumps({"name": name}).encode(), method="POST",
        headers={"x-upload-token": UPLOAD_TOKEN, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(json.loads(resp.read())["seed"])
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return None


def start_run(name: str, hours: float, seed: int | None = None, share: bool = False) -> dict:
    with _lock:
        state = load_state()
        if is_running(state.get("pid")):
            return {"ok": False, "error": "A run is already going -- stop it first."}
        if seed is None:
            # Keep this computer's seed if it already has one (starting again
            # after a restart continues the same search); otherwise take the
            # next from the site; offline, a random one far above the
            # hand-assigned numbers.
            seed = (state.get("seed") or claim_seed(name)
                    or 10_000 + int.from_bytes(os.urandom(2), "big"))
        # Always lowest priority: anything the owner opens gets the CPU first,
        # and the search only uses what's left over.
        nice = ["nice", "-n", "19"] if shutil.which("nice") else []
        with open(LOG_PATH, "w") as log_fh:
            proc = subprocess.Popen(
                nice + [PYTHON, "scripts/run_montecarlo.py", "--overnight",
                        "--hours", str(hours), "--seed", str(seed),
                        "--material", "paperboard_unknown"],
                cwd=ROOT, stdout=log_fh, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        # A sleeping computer pauses the search. On a Mac this keeps it awake
        # for exactly as long as the run lasts (closing a laptop lid still
        # sleeps it; nothing short of settings changes that).
        _children.append(proc)
        if host_kind() == "mac" and shutil.which("caffeinate"):
            _children.append(subprocess.Popen(
                ["caffeinate", "-i", "-w", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
        save_state({"name": name, "hours": hours, "seed": seed, "share": share,
                    "pid": proc.pid, "started_at": time.time()})
        return {"ok": True, "seed": seed, "pid": proc.pid}


def set_share(enabled: bool) -> dict:
    with _lock:
        state = load_state()
        state["share"] = enabled
        save_state(state)
    return {"ok": True}


def stop_run() -> dict:
    with _lock:
        state = load_state()
    pid = state.get("pid")
    if not is_running(pid):
        return {"ok": False, "error": "Nothing is running."}
    try:
        os.killpg(pid, signal.SIGCONT)  # a paused search can't act on the stop until resumed
    except OSError:
        pass
    os.kill(pid, signal.SIGINT)
    return {"ok": True}


def run_quick_test() -> dict:
    cmd = [PYTHON, "scripts/run_montecarlo.py", "--named", "recommended",
           "-n", "50", "--seed", "1", "--material", "paperboard_unknown"]
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "Timed out after 2 minutes -- something is wrong with the setup."}
    output = (result.stdout or "") + (result.stderr or "")
    return {"ok": result.returncode == 0, "output": output.strip()}


def run_upload(name: str) -> dict:
    if not name:
        return {"ok": False, "output": "Type your name first."}
    env = dict(os.environ, UPLOAD_TOKEN=UPLOAD_TOKEN)
    cmd = [PYTHON, "scripts/upload_results.py", "--tag", name, "--url", UPLOAD_URL]
    try:
        # A real overnight run's output can run to hundreds of MB, uploaded in
        # chunks -- generous so a slow connection reads as slow, not broken.
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "Upload timed out -- check your internet connection."}
    output = (result.stdout or "") + (result.stderr or "")
    return {"ok": result.returncode == 0, "output": output.strip()}


# --------------------------------------------------------- sharing the computer

IDLE_AFTER_S = 120   # no keyboard or mouse for this long = they've stepped away
SHARE_DUTY = 0.35    # while they're on it, the search runs this fraction of each second

# GetLastInputInfo through a long-lived PowerShell: starting PowerShell fresh
# for every reading would itself cost about a second of CPU each time.
_WINDOWS_IDLE_PS = r"""
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class GuestIdle {
  [StructLayout(LayoutKind.Sequential)] struct LII { public uint cbSize; public uint dwTime; }
  [DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LII info);
  public static uint Ms() { var l = new LII(); l.cbSize = (uint)Marshal.SizeOf(l); GetLastInputInfo(ref l); return unchecked((uint)Environment.TickCount - l.dwTime); }
}
'@
while ($true) { [Console]::Out.WriteLine([GuestIdle]::Ms()); [Console]::Out.Flush(); Start-Sleep -Seconds 3 }
"""


class _WindowsIdle:
    """Keyboard/mouse idle time on the Windows side, read from inside WSL."""

    def __init__(self):
        self.proc = None
        self.seconds = None
        self.updated = 0.0

    def _pump(self):
        for line in self.proc.stdout:
            try:
                self.seconds = int(line.strip()) / 1000
                self.updated = time.time()
            except ValueError:
                pass

    def read(self) -> float | None:
        if self.proc is None:
            try:
                self.proc = subprocess.Popen(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_IDLE_PS],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL, text=True)
                threading.Thread(target=self._pump, daemon=True).start()
            except OSError:
                self.proc = False
        if not self.proc or time.time() - self.updated > 15:
            return None
        return self.seconds

    def close(self):
        if self.proc:
            self.proc.terminate()


def input_idle_seconds(windows_idle: _WindowsIdle) -> float | None:
    """Seconds since the last keyboard or mouse input, or None where that
    can't be read (no screen, or a Linux desktop without xprintidle)."""
    try:
        if is_wsl():
            return windows_idle.read()
        if host_kind() == "mac":
            out = subprocess.run(["ioreg", "-c", "IOHIDSystem"],
                                 capture_output=True, text=True, timeout=5).stdout
            match = re.search(r'"HIDIdleTime" = (\d+)', out)
            return int(match.group(1)) / 1e9 if match else None
        if os.environ.get("DISPLAY") and shutil.which("xprintidle"):
            out = subprocess.run(["xprintidle"], capture_output=True, text=True, timeout=5).stdout
            return int(out.strip()) / 1000
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


class _OtherCpu:
    """CPU cores in use by everything *except* the search, from /proc -- the
    fallback on Linux when keyboard/mouse idle time isn't readable (a Pi with
    no screen: "someone's using it" then means someone SSH'd in and working)."""

    def __init__(self):
        self.prev = None
        self.tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

    def sample(self, pgid: int) -> float | None:
        try:
            with open("/proc/stat") as f:
                vals = [int(v) for v in f.readline().split()[1:9]]
        except (OSError, ValueError):
            return None
        busy = sum(vals) - vals[3] - vals[4]
        ours = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if int(fields[2]) == pgid:
                ours[int(entry)] = int(fields[11]) + int(fields[12])
        now = time.time()
        prev, self.prev = self.prev, (now, busy, ours)
        if prev is None:
            return None
        t0, busy0, ours0 = prev
        ours_delta = sum(t - ours0[pid] for pid, t in ours.items() if pid in ours0)
        return max(0.0, (busy - busy0 - ours_delta) / self.tick / max(now - t0, 0.1))


class Governor(threading.Thread):
    """Only when the guest ticked "I'll be using this computer": while they're
    on it, pause and resume the whole search (its own process group, so every
    worker at once) each second, leaving it SHARE_DUTY of the time; once
    they've stepped away, let it run flat out. Every run is already low
    priority, which gives whatever they open first claim on the CPU -- this
    adds real headroom on top, and keeps the machine cooler and quieter."""

    def __init__(self):
        super().__init__(daemon=True)
        self.halt = threading.Event()
        self.mode = None  # None, "full", or "sharing"
        self._windows_idle = _WindowsIdle()
        self._other_cpu = _OtherCpu()

    @staticmethod
    def _signal(pid, sig):
        try:
            os.killpg(pid, sig)
        except OSError:
            pass

    def _user_active(self, pid: int) -> bool:
        idle = input_idle_seconds(self._windows_idle)
        if idle is not None:
            return idle < IDLE_AFTER_S
        if is_wsl():
            return True  # can't see Windows activity -- they ticked the box, so assume yes
        cores = self._other_cpu.sample(pid)
        return True if cores is None else cores > 0.75

    def run(self):
        paused = None
        active, checked = True, 0.0
        while not self.halt.is_set():
            reap_children()
            state = load_state()
            pid = state.get("pid")
            if not state.get("share") or not is_running(pid):
                if paused:
                    self._signal(paused, signal.SIGCONT)
                    paused = None
                self.mode = None
                self.halt.wait(2)
                continue
            if time.time() - checked >= 5:
                active = self._user_active(pid)
                checked = time.time()
            if not active:
                if paused:
                    self._signal(paused, signal.SIGCONT)
                    paused = None
                self.mode = "full"
                self.halt.wait(1)
                continue
            self.mode = "sharing"
            self._signal(pid, signal.SIGCONT)
            self.halt.wait(SHARE_DUTY)
            self._signal(pid, signal.SIGSTOP)
            paused = pid
            self.halt.wait(1 - SHARE_DUTY)
        if paused:
            self._signal(paused, signal.SIGCONT)
        self._windows_idle.close()


_governor: Governor | None = None


# ------------------------------------------------------------------- the page

# A raw string: the JavaScript below needs its "\n" escapes to reach the
# browser as backslash-n, not as real line breaks that end a string literal.
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
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
button:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
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
@media (prefers-reduced-motion: reduce){ .dot{animation:none !important} }
.phase-label{font-family:var(--font-display); font-weight:700; font-size:15px}
.seed-tag{
  font-family:var(--font-mono); font-size:12px; color:var(--ink-soft);
  background:var(--ground); border:1px solid var(--line); border-radius:6px;
  padding:2px 8px; margin-left:auto;
}
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
.bar-note{
  display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;
  border-radius:10px; padding:12px 16px; margin-bottom:20px; font-size:13.5px;
}
.bar-note button{white-space:nowrap}
.bar-note .dismiss{background:none; text-decoration:underline; padding:6px 4px; color:inherit}
.shortcut-bar{background:var(--accent-soft); color:var(--accent-ink)}
.safe-banner{background:var(--good-soft); color:var(--ink); margin-bottom:16px}
.safe-banner b{color:var(--good)}
.offline-banner{background:var(--warn-soft); color:var(--ink); display:block}
.offline-banner b{color:var(--warn)}
.check{display:flex; align-items:flex-start; gap:10px; text-transform:none; letter-spacing:0; font-family:var(--font-body); font-weight:600; font-size:14.5px; color:var(--ink); margin:2px 0 4px; cursor:pointer}
.check input{width:18px; height:18px; margin:2px 0 0; accent-color:var(--accent); flex-shrink:0; cursor:pointer}
.check-hint{font-size:13px; color:var(--ink-soft); margin:0 0 14px 28px}
.share-note{font-size:13px; color:var(--ink-soft); margin:-6px 0 14px}
.after-upload{font-size:14px; margin:12px 0 0}
.after-upload a,.footer a{color:var(--accent-ink)}
[hidden]{display:none !important}
</style></head><body>
<div class="wrap">
  <h1>Canoe Hull Search</h1>
  <p class="dek">Type your name, hit start, and leave the computer on. You can close this tab -- the search keeps going either way -- and come back any time to see how it's doing.</p>

  <div class="bar-note offline-banner" id="offline-banner" hidden>
    <b>Can't reach the panel right now.</b> It has probably stopped -- usually because the computer restarted or the window running it was closed. Start it again with the <b>Canoe Hull Search</b> icon on your desktop, or by pasting the setup command again, then click <b>Start run</b> -- it picks up where it left off. This page reconnects on its own.
  </div>

  <div class="bar-note shortcut-bar" id="shortcut-bar">
    <span>Want an icon on your desktop that starts this page again, even after a restart?</span>
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
      <div class="field">
        <label for="seed">Seed (optional)</label>
        <input type="number" id="seed" placeholder="auto" min="0" step="1">
      </div>
    </div>
    <label class="check"><input type="checkbox" id="share"> I'll be using this computer while it runs</label>
    <p class="check-hint">It backs off whenever you're on the computer and speeds back up when you step away -- so it checks fewer hull designs than if you leave it alone. You can change this any time.</p>
    <p class="hint">8 hours is a good overnight default -- it fits the search into whatever window you give it and never runs long. Keep the computer plugged in and don't let it go to sleep; sleep pauses the search. Leave Seed blank -- each computer automatically gets its own number, so none repeat each other's work.</p>
    <div class="actions">
      <button class="btn-secondary" id="btn-test">Test setup (10 sec)</button>
      <button class="btn-primary" id="btn-start">Start run</button>
    </div>
    <div id="test-result" class="result-box" hidden></div>
  </div>

  <div class="card" id="status-card" hidden>
    <div class="bar-note safe-banner" id="safe-banner" hidden>
      <span>&#10003; <b>It's running.</b> <span id="safe-text"></span></span>
      <button class="dismiss" id="btn-safe-dismiss">Got it</button>
    </div>
    <div class="status">
      <span class="dot" id="status-dot"></span>
      <span class="phase-label" id="status-label">--</span>
      <span class="seed-tag" id="seed-tag" hidden></span>
    </div>
    <div class="bar" id="bar-wrap" hidden><div class="bar-fill" id="bar-fill" style="width:0%"></div></div>
    <p class="detail" id="status-detail"></p>
    <p class="share-note" id="share-note" hidden></p>
    <div class="actions">
      <button class="btn-danger" id="btn-stop">Stop</button>
      <button class="btn-primary" id="btn-upload" hidden>Send results to Levi</button>
    </div>
    <div id="upload-result" class="result-box" hidden></div>
    <p class="after-upload" id="after-upload" hidden>Done lending your computer? <a href="https://cuboat.netlify.app/guest#remove" target="_blank" rel="noopener">Here's how to remove it</a> -- it's one command.</p>
    <details>
      <summary>Show raw log</summary>
      <div class="result-box" id="log-box" style="max-height:260px"></div>
    </details>
  </div>

  <p class="footer">Part of a cardboard canoe hull simulator for a college engineering class's lake race. &middot; <a href="https://cuboat.netlify.app/guest#remove" target="_blank" rel="noopener">How to remove it</a></p>
</div>

<script>
const $ = id => document.getElementById(id);
const nameEl = $('name'), hoursEl = $('hours'), seedEl = $('seed'), shareEl = $('share');
const OFFLINE = "Can't reach the panel right now -- it may have stopped.";

function store(kind, key, value) {
  try { return value === undefined ? window[kind].getItem(key) : window[kind].setItem(key, value); }
  catch (e) { return null; }
}

nameEl.value = store('localStorage', 'guestName') || '';
nameEl.addEventListener('input', () => store('localStorage', 'guestName', nameEl.value));
shareEl.addEventListener('change', async () => {
  await api('/api/share', {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({ enabled: shareEl.checked }),
  });
  poll();
});

async function api(path, opts) {
  try {
    const res = await fetch(path, opts);
    return await res.json();
  } catch (e) {
    return { ok: false, offline: true, error: OFFLINE, output: OFFLINE };
  }
}

if (store('localStorage', 'shortcutDismissed')) $('shortcut-bar').hidden = true;

$('btn-safe-dismiss').addEventListener('click', () => {
  store('sessionStorage', 'safeBannerDismissed', '1');
  $('safe-banner').hidden = true;
});

$('btn-shortcut-dismiss').addEventListener('click', () => {
  store('localStorage', 'shortcutDismissed', '1');
  $('shortcut-bar').hidden = true;
});

$('btn-shortcut').addEventListener('click', async () => {
  const btn = $('btn-shortcut');
  btn.disabled = true; btn.textContent = 'Adding...';
  const r = await api('/api/add_shortcut', { method: 'POST' });
  const box = $('shortcut-result');
  box.hidden = false; box.className = 'result-box ' + (r.ok ? 'ok' : 'bad');
  if (r.ok) {
    box.textContent = 'Done! Double-click "Canoe Hull Search" on your desktop any time to get back here -- even after a restart.' + (r.note ? '\n\n' + r.note : '');
    store('localStorage', 'shortcutDismissed', '1');
    $('shortcut-bar').hidden = true;
  } else {
    box.textContent = "Couldn't add it automatically: " + r.error;
  }
  btn.disabled = false; btn.textContent = 'Add to desktop';
});

$('btn-test').addEventListener('click', async () => {
  const btn = $('btn-test');
  btn.disabled = true; btn.textContent = 'Testing...';
  const box = $('test-result');
  box.hidden = false; box.className = 'result-box'; box.textContent = 'Running a quick check...';
  const r = await api('/api/test', { method: 'POST' });
  box.className = 'result-box ' + (r.ok ? 'ok' : 'bad');
  box.textContent = (r.ok ? 'Setup looks good! Safe to start a real run.\n\n' : 'Something is wrong:\n\n') + r.output;
  btn.disabled = false; btn.textContent = 'Test setup (10 sec)';
});

$('btn-start').addEventListener('click', async () => {
  const name = nameEl.value.trim();
  if (!name) { nameEl.focus(); return; }
  const hours = parseFloat(hoursEl.value) || 8;
  const seed = seedEl.value.trim() === '' ? null : parseInt(seedEl.value, 10);
  const btn = $('btn-start');
  btn.disabled = true; btn.textContent = 'Starting...';
  const r = await api('/api/start', {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({ name, hours, seed, share: shareEl.checked }),
  });
  btn.textContent = 'Start run';
  if (!r.ok) { btn.disabled = false; alert(r.error); return; }
  store('sessionStorage', 'safeBannerDismissed', '');
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
  box.hidden = false; box.className = 'result-box'; box.textContent = 'Uploading -- a big run can take a few minutes...';
  const r = await api('/api/upload', {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({ name: nameEl.value.trim() }),
  });
  box.className = 'result-box ' + (r.ok ? 'ok' : 'bad');
  box.textContent = (r.ok ? 'Sent! Thank you for the help.\n\n' : 'Upload failed:\n\n') + r.output;
  $('after-upload').hidden = !r.ok;
  btn.disabled = false; btn.textContent = 'Send results to Levi';
});

const PHASE_LABEL = {
  idle: 'Not started', starting: 'Starting...', preparing: 'Preparing...',
  searching: 'Searching for the best hull', done: 'Finished',
  stopped: 'Stopped', error: 'Something went wrong',
};

let timer = null;
let filledFromState = false;
async function poll() {
  clearTimeout(timer);
  const s = await api('/api/status');
  $('offline-banner').hidden = !s.offline;
  if (s.offline) { timer = setTimeout(poll, 5000); return; }

  // Only once: re-filling on every poll would overwrite what someone is typing.
  if (!filledFromState) {
    if (s.name && !nameEl.value) nameEl.value = s.name;
    if (s.hours) hoursEl.value = s.hours;
    shareEl.checked = !!s.share;
    filledFromState = true;
  }

  $('status-card').hidden = s.phase === 'idle' && !s.started_at;
  $('safe-banner').hidden = !s.running || !!store('sessionStorage', 'safeBannerDismissed');
  $('safe-text').textContent = s.wsl
    ? 'Safe to close this browser tab -- but leave the Ubuntu window open (minimizing is fine), since closing it stops the search.'
    : 'Safe to close this browser tab, or your whole browser -- the search keeps going on its own. Come back to ' + location.host + ' any time to check on it.';
  $('status-dot').className = 'dot ' + s.phase;
  $('status-label').textContent = PHASE_LABEL[s.phase] || s.phase;
  $('seed-tag').hidden = s.seed == null;
  if (s.seed != null) $('seed-tag').textContent = 'seed ' + s.seed;
  $('status-detail').textContent = s.detail || '';
  $('share-note').hidden = !s.sharing;
  $('share-note').textContent = s.sharing === 'sharing'
    ? "You're using the computer, so it's running lighter for now."
    : "You've stepped away, so it's running at full speed.";
  $('bar-wrap').hidden = s.progress == null;
  if (s.progress != null) $('bar-fill').style.width = s.progress + '%';
  $('btn-stop').hidden = !s.running;
  $('btn-upload').hidden = !(s.phase === 'done' || s.phase === 'stopped');
  $('log-box').textContent = (s.log_tail || []).join('\n') || '(nothing yet)';
  $('btn-start').disabled = s.running;
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
        except ValueError:
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
            seed = None
            if data.get("seed") not in (None, ""):
                try:
                    seed = max(0, min(2**31 - 1, int(data["seed"])))
                except (TypeError, ValueError):
                    seed = None
            self._send_json(start_run(name, hours, seed, bool(data.get("share"))))
        elif self.path == "/api/share":
            self._send_json(set_share(bool(self._read_json().get("enabled"))))
        elif self.path == "/api/stop":
            self._send_json(stop_run())
        elif self.path == "/api/test":
            self._send_json(run_quick_test())
        elif self.path == "/api/upload":
            data = self._read_json()
            self._send_json(run_upload(str(data.get("name", "")).strip()[:40]))
        elif self.path == "/api/add_shortcut":
            self._send_json(create_desktop_shortcut())
        elif self.path == "/api/shutdown":
            # Only from this machine: it's how a newer copy replaces this one.
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                self._send_json({"ok": False}, 403)
                return
            self._send_json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()


# ------------------------------------------------------------------ terminal

BOAT = [
    " " * 14 + "o" + " " * 16 + "o",
    " " * 13 + "/|\\___" + " " * 11 + "/|\\___",
    " " * 6 + "_______/_\\____\\_________/_\\____\\_______",
    " " * 6 + "\\" + " " * 37 + "/",
    " " * 7 + "\\__________  C U B O A T  __________/",
]
WAVE = "~~~^~~  ~~~~ ~^~~~  ~~ "


def _color_on() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _paint(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _color_on() else s


def _waves(frame: int) -> list[str]:
    n = len(WAVE)
    unit = WAVE * 6
    return ["  " + unit[frame % n:][:48], "    " + unit[(n - frame % n) % n:][:44]]


def boat(animate: bool = True) -> None:
    print()
    for line in BOAT:
        print(_paint("33", line))
    for line in _waves(0):
        print(_paint("36", line))
    sys.stdout.flush()
    if animate and _color_on():
        for frame in range(1, 18):
            time.sleep(0.07)
            sys.stdout.write("\033[2A")
            for line in _waves(frame):
                sys.stdout.write("\r" + _paint("36", line) + "\n")
            sys.stdout.flush()


def show_banner(*, already: bool, background: bool) -> None:
    boat()
    url = f"http://localhost:{PORT}"
    head = ("The hull search panel was already running." if already
            else "The hull search panel is up and running.")
    lines = ["", "  " + _paint("1", head), ""]
    if can_open_browser():
        lines += [f"  Open:  {_paint('1;36', url)}",
                  "  Opening it in your browser now -- if nothing pops up, type that address into any browser."]
    else:
        ip = lan_ip()
        remote = f"http://{ip}:{PORT}" if ip else f"http://<this computer's IP address>:{PORT}"
        lines += ["  No screen detected here, so no browser will pop up on this computer.",
                  f"  On your phone or laptop, on the same Wi-Fi, open:  {_paint('1;36', remote)}"]
    lines.append("")
    if is_wsl():
        lines += ["  " + _paint("1;33", "Leave this Ubuntu window open while it runs (minimizing is fine)."),
                  "  On Windows, closing it can shut Linux down and stop the search."]
    elif background or already:
        lines.append("  It runs in the background, so you can close this window.")
    else:
        lines.append("  Leave this window open while it runs -- closing it stops the panel.")
    print("\n".join(lines) + "\n", flush=True)


def panel_status() -> dict | None:
    """What answers on the panel's port: its status, {"_foreign": True} if
    some other program holds the port, or None if nothing is listening."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=2) as resp:
            raw = resp.read()
    except urllib.error.HTTPError:
        return {"_foreign": True}
    except (urllib.error.URLError, OSError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return {"_foreign": True}
    return data if isinstance(data, dict) and "phase" in data else {"_foreign": True}


def _wait_for(predicate, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


def replace_outdated_panel() -> bool:
    """Stop a panel still running from an older copy of this script, so
    re-running the setup command (which pulls updates) serves the new code
    instead of quietly reopening the old page."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/shutdown", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=2).read()
    except (urllib.error.URLError, OSError):
        # Copies from before /api/shutdown existed answer 404 here.
        for pid in _old_panel_pids():
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    return _wait_for(lambda: panel_status() is None, 5)


def _old_panel_pids() -> list[int]:
    """Find an old panel's process: by port where lsof exists (always on a
    Mac), otherwise by command line and working folder through /proc --
    Raspberry Pi OS Lite doesn't ship lsof."""
    found = []
    if shutil.which("lsof"):
        pids = subprocess.run(["lsof", "-t", f"-iTCP:{PORT}", "-sTCP:LISTEN"],
                              capture_output=True, text=True).stdout.split()
        for pid in pids:
            command = subprocess.run(["ps", "-o", "command=", "-p", pid],
                                     capture_output=True, text=True).stdout
            if "guest_panel.py" in command:
                found.append(int(pid))
    elif Path("/proc").is_dir():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes()
                cwd = Path(os.readlink(entry / "cwd"))
            except OSError:
                continue
            if b"guest_panel.py" in cmdline and cwd == ROOT:
                found.append(int(entry.name))
    return [pid for pid in found if pid != os.getpid()]


def port_taken_message() -> str:
    return (f"\n  Port {PORT} is being used by a different program, so the panel can't start.\n"
            f"  Close that program, or pick another port, e.g.:\n"
            f"    GUEST_PANEL_PORT=8421 {PYTHON} scripts/guest_panel.py start\n")


def cmd_start() -> int:
    status = panel_status()
    if status and status.get("_foreign"):
        print(port_taken_message())
        return 1
    if status and status.get("version") != VERSION and replace_outdated_panel():
        status = None
    already = status is not None
    if not already:
        with open(PANEL_LOG_PATH, "a") as log:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve"],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        if not _wait_for(lambda: panel_status() is not None, 15):
            print("\n  The panel didn't start. The end of panel.log says:\n")
            print("\n".join("    " + line for line in tail_lines(PANEL_LOG_PATH, 15)) + "\n")
            return 1
    show_banner(already=already, background=True)
    open_browser(f"http://localhost:{PORT}")
    return 0


def serve(quiet: bool) -> int:
    status = panel_status()
    if status and status.get("_foreign"):
        print(port_taken_message())
        return 1
    if status and (status.get("version") == VERSION or not replace_outdated_panel()):
        if not quiet:
            show_banner(already=True, background=False)
            open_browser(f"http://localhost:{PORT}")
        return 0

    host = bind_host()
    try:
        server = ThreadingHTTPServer((host, PORT), Handler)
    except OSError as e:
        print(f"  Couldn't start the panel on port {PORT}: {e}", flush=True)
        return 1
    PANEL_PID_PATH.write_text(str(os.getpid()))

    # If an earlier panel died mid-pause (killed outright), its search would
    # otherwise stay frozen forever.
    pid = load_state().get("pid")
    if is_running(pid):
        Governor._signal(pid, signal.SIGCONT)
    global _governor
    _governor = Governor()
    _governor.start()
    signal.signal(signal.SIGTERM,
                  lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())

    if quiet:
        print(f"panel {VERSION} listening on {host}:{PORT}", flush=True)
    else:
        show_banner(already=False, background=False)
        open_browser(f"http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Panel stopped. A search that was running keeps going.", flush=True)
    finally:
        _governor.halt.set()
        _governor.join(3)
        server.server_close()
        try:
            PANEL_PID_PATH.unlink()
        except OSError:
            pass
    return 0


def cmd_uninstall() -> int:
    boat(animate=False)
    if ROOT != GUEST_INSTALL:
        print(f"\n  This only removes the standard guest install at {GUEST_INSTALL}.\n"
              f"  This copy lives at {ROOT}, so it's left alone -- delete it yourself if you mean to.\n")
        return 1
    print(f"\n  This removes the hull search from this computer:\n"
          f"    - stops the panel, and a search if one is running\n"
          f"    - deletes the desktop icon, if you made one\n"
          f"    - deletes {ROOT}\n\n"
          f"  If you haven't clicked \"Send results to Levi\" yet, do that first -- this deletes them.\n")
    try:
        answer = input('  Type "yes" to remove it: ')
    except EOFError:
        answer = ""
    if answer.strip().lower() != "yes":
        print("  Nothing removed.\n")
        return 1

    # The panel first: while it runs, its governor may be pausing the search,
    # and a paused process can't act on a stop signal.
    status = panel_status()
    if status and not status.get("_foreign"):
        print("  Stopping the panel...")
        replace_outdated_panel()

    pid = load_state().get("pid")
    if is_running(pid):
        print("  Stopping the search...")
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 3)):
            try:
                os.killpg(pid, signal.SIGCONT)
                os.killpg(pid, sig)  # its own process group, workers included
            except OSError:
                pass
            if _wait_for(lambda: not is_running(pid), wait):
                break

    desktop = desktop_path()
    if desktop:
        remove_desktop_shortcuts(desktop)
    shutil.rmtree(ROOT, ignore_errors=True)

    print("\n  " + _paint("1", "Removed. Thanks for lending your computer to the boat!"))
    if is_wsl():
        print("  To remove Linux itself too, run this in Windows PowerShell:  wsl --unregister Ubuntu")
    print()
    return 0


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "start":
        return cmd_start()
    if command == "run":
        return serve(quiet=False)
    if command == "serve":
        return serve(quiet=True)
    if command == "uninstall":
        return cmd_uninstall()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
