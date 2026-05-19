# Genesiss

A local text-to-CAD suite. You describe a part in plain English; a finetuned LLM running through Ollama on your machine emits CADQuery code; the result is exec'd and rendered in a 3D viewer. One CLI boots the whole stack.

## Goal

Make text-to-CAD feel like text-to-image — fast, local, hackable. No cloud round trip, no API key, no per-part token bill. Everything runs on the user's own hardware:

- **Local inference.** Ollama serves the model. No data leaves the box.
- **Editable output.** The generated CADQuery script is shown in a Monaco editor — you can tweak it and re-run, instead of being stuck with a black-box mesh.
- **Real CAD.** CADQuery emits parametric solid models (STEP), not just triangle soup. The 3D viewer is just a preview; the artifact is something you can drop into FreeCAD or a manufacturing pipeline.
- **One CLI to rule it.** `genesiss` boots Ollama-aware orchestration, the Python bridge, and the Electron frontend (or a Textual TUI if you'd rather stay in the terminal).

## Variants

Four sizes covering hobbyist hardware up to a single H100. All four are finetunes of openly-available bases on the [Text-to-CadQuery](https://github.com/Text-to-CadQuery/Text-to-CadQuery) dataset (~170k NL → CADQuery pairs).

| Variant         | Base model      | Trains on               |
| --------------- | --------------- | ----------------------- |
| `genesiss-4b`   | Qwen 3.5 4B     | Colab T4/L4, Kaggle T4×2 |
| `genesiss-9b`   | Qwen 3.5 9B     | Colab L4/A100, Kaggle T4×2 |
| `genesiss-20b`  | gpt-oss-20b     | Colab A100/H100, Kaggle T4×2 (QLoRA) |
| `genesiss-27b`  | Qwen 3.5 27B    | Colab H100 only         |

## Architecture

```
            ┌─────────────────────────────┐
            │     genesiss (CLI)          │
            │  Typer · Rich · Textual     │
            └────────────┬────────────────┘
                         │ launches
                         ▼
           ┌──────────────────────────────┐
           │   orchestrator.launch()      │
           │   - check ollama reachable   │
           │   - start FastAPI bridge     │
           │   - spawn Electron OR TUI    │
           └──────┬────────────────┬──────┘
                  │                │
                  ▼                ▼
       ┌─────────────────┐   ┌─────────────────┐
       │  FastAPI bridge │   │  Textual TUI    │  (--headless)
       │  127.0.0.1:8765 │   │  in the same    │
       │                 │   │  process tree   │
       │  /generate      │   │                 │
       │  /exec          │   │  talks to the   │
       │  /models        │   │  bridge over    │
       │  /ws/generate   │   │  HTTP           │
       └──┬──────────────┘   └─────────────────┘
          │
          │ HTTP                                      ┌─────────────────────────┐
          ▼                                           │  Electron (GUI mode)    │
   ┌─────────────┐         ┌────────────────┐         │  React · Vite · TS      │
   │ Ollama HTTP │◀────────│  llm.OllamaCli │         │  three.js · Monaco      │
   │ 11434       │         │  (httpx)       │         │  fetch → bridge         │
   └─────────────┘         └────────────────┘         └─────────────────────────┘
          ▲
          │ Modelfile (training/modelfiles/*.Modelfile)
          │ uses GGUF Q4_K_M produced by training notebook
```

### Request flow

```
user prompt
   │  (Electron renderer  or  Textual TUI)
   ▼
POST /generate                  ── bridge
   │
   ▼  build_messages(system + user)
ollama /api/chat                ── local Ollama on :11434
   │
   ▼  raw CADQuery python
POST /exec                      ── bridge
   │
   ▼  AST deny-list (validator.py)
   ▼  exec in fresh namespace; expect `result`
   ▼  cadquery.exporters → STL + STEP (base64)
   │
   ▼  STL bytes → @react-three/fiber → on-screen preview
   ▼  STEP bytes → download / manufacturing
```

## Layout

```
genesiss/        # Python backend + CLI
  cli.py          - Typer entry: `genesiss`, --headless / --gui, subcommands
  orchestrator.py - boots Ollama check + bridge + Electron / TUI
  config.py       - persistent config at platformdirs.user_config_dir/genesiss/
  tui.py          - Textual TUI (headless mode)
  server/         - FastAPI bridge: /generate, /exec, /models, /ws/generate
  llm/            - Ollama HTTP client, variant registry, prompt builder
  cad/            - AST validator + CADQuery executor → STL/STEP base64

frontend/        # Electron + React + Vite + TypeScript
  src/main/       - Electron main process, opens single window
  src/preload/    - exposes window.genesiss.bridgeUrl to renderer
  src/renderer/   - React app: Monaco code view + three.js STL preview

training/        # Unsloth finetuning
  shared/         - dataset loader, chat-template wiring, async Hub checkpointing
  colab/          - 4 notebooks, one per variant, ready to open in Colab
  kaggle/         - 4 notebooks, one per variant, ready to open in Kaggle
  modelfiles/     - Ollama Modelfile per variant (used after training)
  _build_notebooks.py - regenerates the .ipynb files from the templates

docs/            - architecture.md, training.md
```

## Install

Prereqs: [uv](https://docs.astral.sh/uv/), [pnpm](https://pnpm.io/), and [Ollama](https://ollama.com/) on `PATH`.

### Install `genesiss` system-wide

`uv tool install` creates an isolated environment for the CLI and puts the `genesiss` executable on your `PATH` — so you can run it from anywhere on your machine, not just inside a checkout of this repo.

```bash
# Straight from GitHub (no clone needed)
uv tool install git+https://github.com/numinousmuses/genesiss

# …or from a local clone
git clone https://github.com/numinousmuses/genesiss && cd genesiss
uv tool install .

# Make sure ~/.local/bin (where uv puts tool shims) is on PATH
uv tool update-shell
exec $SHELL -l

# Sanity check
genesiss --help
```

To upgrade later: `uv tool upgrade genesiss`. To remove: `uv tool uninstall genesiss`.

### Frontend (Electron) one-time setup

The GUI is an Electron app that lives alongside the CLI. The system-installed `genesiss` looks for it relative to a clone of this repo, so you still want the repo on disk for GUI mode:

```bash
git clone https://github.com/numinousmuses/genesiss ~/code/genesiss
cd ~/code/genesiss/frontend && pnpm install
# tell genesiss where the frontend lives
genesiss config set frontend-dir ~/code/genesiss/frontend
```

If you only ever use the headless TUI, you can skip the frontend setup entirely.

### Run it

```bash
# 1. Make sure Ollama is up
ollama serve &

# 2. Pull a trained variant
genesiss models pull genesiss-4b

# 3. Launch (GUI by default)
genesiss

# Headless TUI
genesiss --headless

# Make headless the default
genesiss config set default-mode headless
```

## Develop

For hacking on the codebase itself, work inside a clone with `uv sync` instead of `uv tool install` — that gives you an editable install and dev deps (pytest, ruff, mypy).

```bash
git clone https://github.com/numinousmuses/genesiss && cd genesiss
uv sync                          # creates .venv with deps + dev tools
(cd frontend && pnpm install)
uv run genesiss --help           # runs the in-repo source
```

`uv run <cmd>` auto-activates `.venv` for that command. Equivalent shell-style: `source .venv/bin/activate && genesiss …`.

## Training

Each variant has Colab and Kaggle notebooks under `training/colab/` and `training/kaggle/`. All eight resume from the latest checkpoint on the HF Hub if one exists, and async-push new checkpoints during training so a notebook timeout doesn't lose work — uploads run on a background thread via `HfApi.upload_folder(run_as_future=True)` while the GPU keeps training.

See `docs/training.md` for the per-family recipe (Qwen 3.5 → 16-bit LoRA; gpt-oss-20b → 4-bit + MXFP4 MoE).

## Status

Scaffold-stage. Backend, frontend, and notebooks compile / parse cleanly but no finetune has been pushed yet, so `genesiss models pull` won't return anything real until a training run completes.
