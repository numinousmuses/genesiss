"""`genesiss` CLI.

Default behavior is determined by config.default_mode (initial default: "gui").
Flags override config:
    --headless   force TUI
    --gui        force Electron
"""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from genesiss import config as cfg_mod
from genesiss.config import Config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Local text-to-CAD suite.",
)
config_app = typer.Typer(help="Inspect or modify persistent config.")
models_app = typer.Typer(help="Manage local Ollama models.")
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")

console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    headless: bool = typer.Option(False, "--headless", help="Force terminal TUI."),
    gui: bool = typer.Option(False, "--gui", help="Force Electron GUI."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the default model variant."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if headless and gui:
        raise typer.BadParameter("--headless and --gui are mutually exclusive.")
    cfg = cfg_mod.load()
    if model:
        cfg = Config(**{**cfg.__dict__, "model": model})
    mode = "headless" if headless else "gui" if gui else cfg.default_mode

    from genesiss.orchestrator import launch

    launch(cfg, mode=mode)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Override bind host."),
    port: Optional[int] = typer.Option(None, help="Override bind port."),
) -> None:
    """Run just the FastAPI bridge — useful for dev against an external Electron."""
    cfg = cfg_mod.load()
    if host:
        cfg = Config(**{**cfg.__dict__, "host": host})
    if port:
        cfg = Config(**{**cfg.__dict__, "port": port})
    from genesiss.server.app import run

    run(cfg)


@config_app.command("show")
def config_show() -> None:
    cfg = cfg_mod.load()
    table = Table(title="genesiss config", show_header=True)
    table.add_column("key", style="cyan")
    table.add_column("value")
    for k, v in cfg.__dict__.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"[dim]path: {cfg_mod.config_path()}[/dim]")


@config_app.command("set")
def config_set(key: str = typer.Argument(...), value: str = typer.Argument(...)) -> None:
    try:
        new = cfg_mod.set_value(key, value)
    except KeyError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(f"[green]set[/green] {key} = {getattr(new, key)}")


@models_app.command("list")
def models_list() -> None:
    """List models known to the registry and which are pulled in Ollama."""
    from genesiss.llm.models import REGISTRY
    from genesiss.llm.ollama import OllamaClient

    cfg = cfg_mod.load()
    pulled = set()
    try:
        pulled = {m.name for m in OllamaClient(cfg.ollama_host).list_sync()}
    except Exception as e:  # noqa: BLE001 — ollama may be down; we want to keep going
        console.print(f"[yellow]ollama unreachable: {e}[/yellow]")

    table = Table(show_header=True)
    table.add_column("variant", style="cyan")
    table.add_column("base")
    table.add_column("ollama tag")
    table.add_column("pulled")
    for v in REGISTRY.values():
        table.add_row(v.name, v.base, v.ollama_tag, "✓" if v.ollama_tag in pulled else "·")
    console.print(table)


@models_app.command("pull")
def models_pull(variant: str = typer.Argument(...)) -> None:
    """Pull a variant's Ollama image."""
    from genesiss.llm.models import REGISTRY
    from genesiss.llm.ollama import OllamaClient

    if variant not in REGISTRY:
        raise typer.BadParameter(f"unknown variant: {variant}. Try: {list(REGISTRY)}")
    cfg = cfg_mod.load()
    client = OllamaClient(cfg.ollama_host)
    for evt in client.pull_sync(REGISTRY[variant].ollama_tag):
        console.print(evt)


@models_app.command("use")
def models_use(variant: str = typer.Argument(...)) -> None:
    """Set a variant as the persistent default."""
    from genesiss.llm.models import REGISTRY

    if variant not in REGISTRY:
        raise typer.BadParameter(f"unknown variant: {variant}. Try: {list(REGISTRY)}")
    cfg_mod.set_value("model", variant)
    console.print(f"[green]default model[/green] → {variant}")


if __name__ == "__main__":
    app()
