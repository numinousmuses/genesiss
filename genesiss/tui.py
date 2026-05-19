"""Textual-based headless interface.

Layout:
    +----------------------------------------------+
    |  prompt input  [ submit ]                    |
    +----------------------------------------------+
    |  generated cadquery code (read-only)         |
    +----------------------------------------------+
    |  exec output / errors                        |
    +----------------------------------------------+
"""
from __future__ import annotations

import asyncio

import httpx
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from genesiss.config import Config


class GenesissTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #prompt { dock: top; height: 3; }
    #code { height: 1fr; border: solid green; }
    #log { height: 30%; border: solid yellow; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+l", "clear", "Clear")]

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.bridge = f"http://{cfg.host}:{cfg.port}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="Describe a part…  (Enter to generate)", id="prompt")
        yield Vertical(
            Static("# cadquery output will appear here", id="code"),
            RichLog(id="log", highlight=True, markup=True),
        )
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        self.query_one("#log", RichLog).write(f"[cyan]> {prompt}[/cyan]")
        event.input.value = ""
        asyncio.create_task(self._generate(prompt))

    async def _generate(self, prompt: str) -> None:
        log = self.query_one("#log", RichLog)
        code_view = self.query_one("#code", Static)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{self.bridge}/generate",
                    json={"prompt": prompt, "model": self.cfg.model},
                )
                r.raise_for_status()
                data = r.json()
            code_view.update(f"```python\n{data['code']}\n```")
            log.write(f"[green]ok[/green] · {len(data['code'])} chars")
        except Exception as e:  # noqa: BLE001
            log.write(f"[red]error[/red] {e}")

    def action_clear(self) -> None:
        self.query_one("#log", RichLog).clear()
        self.query_one("#code", Static).update("# cleared")
