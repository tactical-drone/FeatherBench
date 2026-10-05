"""
rundir: run directories (runs/<name>/) for sweeps and searches, their lock,
heartbeat, STOP file and status, plus background launching.

    rd = RunDir.create("runs/s5a", "evolve", manifest)    # E_EXISTS / E_LOCKED
    rd.append("evals.jsonl", {...}); rd.write_json("leaderboard.json", lb)
    rd.write_progress(state="running", generation=3)       # heartbeat (call at least every 5 s)
    if rd.stop_requested(): ...
    rd.close("done")

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from .errors import RsiError
from .io import now_iso, to_json

STALE_S = 120.0
FINAL_STATES = ("stopped", "budget", "done", "crashed")


def atomic_write_json(path, obj, *, ascii=True, pretty=True):
    from nncore.configio import atomic_write_text
    try:
        atomic_write_text(path, to_json(obj, pretty=pretty, ascii=ascii) + "\n")
    except OSError as e:
        raise RsiError(f"cannot write {path}: {e}", "E_IO") from None


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    """Records of a JSONL file; a torn (partially written) line is skipped."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return out


def pid_alive(pid):
    if not pid:
        return False
    pid = int(pid)
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie child of ours counts as dead
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ")[-1][:1] != "Z"
    except OSError:
        return True


def resolve_target(target, store_root):
    """NAME -> <store>/NAME; an existing path -> that path. E_NOT_FOUND if neither exists."""
    p = Path(target)
    for cand in (p, Path(store_root) / target):
        if cand.is_dir():
            return cand
    raise RsiError(f"no run dir '{target}' (looked in . and {store_root})", "E_NOT_FOUND", value=str(target),
                   hint="python -m rsi runs list")


class RunDir:
    def __init__(self, path, locked=False):
        self.path = Path(path)
        self.locked = locked
        self._beat = 0.0
        self._progress = read_json(self.path / "progress.json", {}) or {}

    @property
    def name(self):
        return self.path.name

    @classmethod
    def create(cls, path, kind, manifest, *, force=False):
        """New run dir with manifest.json; takes the lock. E_EXISTS if a manifest is already
        there (unless force), E_LOCKED if a live process holds it."""
        path = Path(path)
        if (path / "manifest.json").exists():
            if not force:
                raise RsiError(f"run dir {path} exists", "E_EXISTS", value=str(path),
                               hint="pick another --name, pass --force, or --resume it")
            _check_lock(path)
            keep = {"log.txt", "stdout.json", "progress.json", "launch.json"} if os.environ.get("RSI_BG_CHILD") else set()
            for p in path.iterdir():
                if p.name in keep:
                    continue
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RsiError(f"cannot create {path}: {e}", "E_IO") from None
        rd = cls(path)
        rd._lock()
        rd.write_json("manifest.json", {"format": "rsi/manifest", "version": 1, "kind": kind, "name": path.name,
                                        "created": now_iso(), **manifest})
        rd.write_progress(state="running", kind=kind, name=path.name, started=now_iso())
        return rd

    @classmethod
    def open(cls, path, *, lock):
        path = Path(path)
        if not (path / "manifest.json").exists():
            raise RsiError(f"{path} is not a run dir (no manifest.json)", "E_NOT_FOUND", value=str(path))
        rd = cls(path)
        if lock:
            rd._lock()
        return rd

    def _lock(self):
        lock = self.path / "lock"
        for _ in range(2):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    json.dump({"pid": os.getpid(), "host": socket.gethostname(), "created": now_iso()}, f)
                self.locked = True
                return
            except FileExistsError:
                _check_lock(self.path)
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
        raise RsiError(f"cannot lock {self.path}", "E_LOCKED", value=str(self.path))

    def manifest(self):
        return read_json(self.path / "manifest.json", {})

    def append(self, name, obj):
        try:
            with open(self.path / name, "a", encoding="utf-8", newline="\n") as f:
                f.write(to_json(obj) + "\n")
        except OSError as e:
            raise RsiError(f"cannot append to {self.path / name}: {e}", "E_IO") from None

    def write_json(self, name, obj):
        atomic_write_json(self.path / name, obj)

    def write_progress(self, **fields):
        """Merge fields into progress.json with a fresh heartbeat and this pid."""
        self._progress.update(fields)
        self._progress.update(pid=os.getpid(), heartbeat=time.time(), heartbeat_iso=now_iso())
        self.write_json("progress.json", self._progress)
        self._beat = time.time()

    def heartbeat(self, every=2.0):
        """write_progress() if the last write is older than `every` seconds."""
        if time.time() - self._beat >= every:
            self.write_progress()

    def stop_requested(self):
        return (self.path / "STOP").exists()

    def stop_mode(self):
        """None, 'graceful' or 'now' (from the STOP file)."""
        f = self.path / "STOP"
        if not f.exists():
            return None
        try:
            return "now" if "now" in f.read_text(encoding="utf-8") else "graceful"
        except OSError:
            return "graceful"

    def close(self, state):
        self.write_progress(state=state, finished=now_iso())
        try:
            (self.path / "STOP").unlink()
        except FileNotFoundError:
            pass
        if self.locked:
            try:
                (self.path / "lock").unlink()
            except FileNotFoundError:
                pass
            self.locked = False


def _check_lock(path):
    """E_LOCKED if the lock's pid is alive and the heartbeat is fresh."""
    info = read_json(Path(path) / "lock")
    if not info:
        return
    prog = read_json(Path(path) / "progress.json", {}) or {}
    fresh = time.time() - float(prog.get("heartbeat") or 0) < STALE_S
    if pid_alive(info.get("pid")) and info.get("pid") != os.getpid() and fresh:
        raise RsiError(f"{path} is locked by live pid {info.get('pid')}", "E_LOCKED", value=str(path),
                       hint=f"python -m rsi stop {Path(path).name}")


# ---------------------------------------------------------------- status
def read_status(path, history=10):
    """The status doc (4.8) from progress.json, with alive / heartbeat_age_s / stale detection."""
    path = Path(path)
    man = read_json(path / "manifest.json", {}) or {}
    prog = read_json(path / "progress.json", {}) or {}
    launch = read_json(path / "launch.json", {}) or {}
    pid = prog.get("pid") or launch.get("pid")
    alive = pid_alive(pid)
    hb = prog.get("heartbeat") or launch.get("t")
    age = round(time.time() - float(hb), 1) if hb else None
    state = prog.get("state") or ("running" if alive else "crashed")
    if (path / "STOP").exists() and state == "running":
        state = "stopping"
    if state in ("running", "stopping") and (not alive or (age is not None and age > STALE_S)):
        state = "stale"
    name = path.name
    out = {"name": name, "dir": str(path), "kind": man.get("kind") or prog.get("kind"), "state": state, "pid": pid,
           "alive": alive, "heartbeat_age_s": age}
    out.update({k: v for k, v in prog.items() if k not in out and k not in ("heartbeat", "heartbeat_iso")})
    hist = read_jsonl(path / "generations.jsonl")
    if hist or out.get("kind") == "evolve":
        out["history"] = hist[-history:] if history else []
    files = {"log": str(path / "log.txt")}
    for key, fn in (("best_settings", "best.settings.json"), ("leaderboard", "leaderboard.json"),
                    ("result", "result.json"), ("stdout", "stdout.json")):
        if (path / fn).exists():
            files[key] = str(path / fn)
    out["files"] = files
    out["next"] = (f"python -m rsi wait {name} --timeout 540" if state in ("running", "stopping")
                   else f"python -m rsi leaderboard {name}" if state in FINAL_STATES
                   else f"python -m rsi {out.get('kind') or 'evolve'} --resume {name}")
    return out


def request_stop(path, now=False):
    (Path(path) / "STOP").write_text("now\n" if now else "graceful\n", encoding="utf-8")


# ---------------------------------------------------------------- background
def strip_flags(argv, flags_no_value=(), flags_with_value=()):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        head = a.split("=", 1)[0]
        if head in flags_no_value:
            continue
        if head in flags_with_value:
            skip = "=" not in a
            continue
        out.append(a)
    return out


def launch_background(argv, path, command):
    """Re-exec `python -m rsi <argv>` detached, stdout -> DIR/stdout.json, stderr -> DIR/log.txt.
    Returns the rsi/background@1 result."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", RSI_BG_CHILD="1")
    from nncore.registry import REPO_ROOT
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    kw = {}
    if os.name == "nt":  # no DETACHED_PROCESS: every spawned worker would open a console window
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    with open(path / "stdout.json", "wb") as out, open(path / "log.txt", "ab") as err:
        p = subprocess.Popen([sys.executable, "-m", "rsi", *argv], stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                             env=env, cwd=os.getcwd(), **kw)
    atomic_write_json(path / "launch.json", {"pid": p.pid, "t": time.time(), "argv": argv, "command": command})
    name = path.name
    return {"name": name, "dir": str(path), "pid": p.pid, "state": "running", "log": str(path / "log.txt"),
            "next": {"status": f"python -m rsi status {name}", "wait": f"python -m rsi wait {name} --timeout 540",
                     "stop": f"python -m rsi stop {name}"}}
