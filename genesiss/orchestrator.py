"""Spins up the pieces a `genesiss` invocation needs.

GUI mode:    start bridge server (in-process or subprocess), launch Electron, wait.
Headless:    start bridge server in a thread, run the Textual TUI in the foreground.

We assume Ollama is already running on the user's machine. We check; if not, we
print a hint and fail fast rather than trying to start it (Ollama is a system
service the user controls).
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import httpx
from rich.console import Console

from genesiss.config import Config

console = Console()
Mode = Literal["gui", "headless"]


def _check_ollama(cfg: Config) -> None:
    try:
        r = httpx.get(f"{cfg.ollama_host}/api/version", timeout=2.0)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        console.print(
            f"[red]Ollama unreachable at {cfg.ollama_host}[/red]\n"
            "Install/start ollama: https://ollama.com  →  then `ollama serve`."
        )
        raise SystemExit(2) from e


def _start_bridge_in_thread(cfg: Config) -> threading.Thread:
    from genesiss.server.app import run

    t = threading.Thread(target=run, args=(cfg,), daemon=True, name="genesiss-bridge")
    t.start()
    # Wait until /health responds.
    deadline = time.monotonic() + 10
    url = f"http://{cfg.host}:{cfg.port}/health"
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return t
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise SystemExit("bridge failed to come up within 10s")


def _frontend_dir() -> Path:
    here = Path(__file__).resolve().parents[1]
    return here / "frontend"


def _spawn_electron(cfg: Config) -> subprocess.Popen[bytes]:
    fdir = _frontend_dir()
    pkg = fdir / "package.json"
    if not pkg.exists():
        raise SystemExit(f"frontend not found at {fdir}. Did you `pnpm install` inside frontend/?")

    runner = shutil.which("pnpm") or shutil.which("npm")
    if runner is None:
        raise SystemExit("Neither pnpm nor npm is on PATH; can't launch Electron.")

    env = os.environ.copy()
    env["GENESISS_BRIDGE_URL"] = f"http://{cfg.host}:{cfg.port}"
    return subprocess.Popen([runner, "run", "dev"], cwd=fdir, env=env)


def launch(cfg: Config, mode: Mode) -> None:
    _check_ollama(cfg)

    if mode == "headless":
        _start_bridge_in_thread(cfg)
        from genesiss.tui import GenesissTUI

        GenesissTUI(cfg).run()
        return

    # GUI mode
    _start_bridge_in_thread(cfg)
    proc = _spawn_electron(cfg)
    try:
        rc = proc.wait()
        sys.exit(rc)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
