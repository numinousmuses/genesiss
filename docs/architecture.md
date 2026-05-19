# Architecture

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

## Pieces

### CLI (`genesiss.cli`)
Typer-based. Top-level callback handles the `--headless` / `--gui` choice and falls back to config. Subcommands: `config`, `models`, `serve`.

### Config (`genesiss.config`)
Frozen dataclass persisted at `platformdirs.user_config_dir("genesiss")/config.toml`. Env vars (`GENESISS_*`, `OLLAMA_HOST`) override file values; CLI flags override env.

### Orchestrator (`genesiss.orchestrator`)
- Health-checks Ollama and bails early with a hint if it's not running (we don't try to start system services).
- Starts the FastAPI bridge in a daemon thread (so SIGINT to the CLI cleanly tears it down).
- In GUI mode, spawns `pnpm/npm run dev` inside `frontend/`, passing `GENESISS_BRIDGE_URL` so the Electron preload picks it up via `additionalArguments`.
- In headless mode, runs the Textual TUI in the foreground.

### Bridge (`genesiss.server`)
FastAPI. Routes:
- `GET /health` — used by the orchestrator to know the bridge came up.
- `GET /models` — returns the variant registry + the configured default.
- `POST /generate` — synchronous one-shot.
- `POST /exec` — validate then exec a CADQuery script. Returns STL + STEP, both base64.
- `WS /ws/generate` — token-by-token streaming for the UI.

### LLM client (`genesiss.llm`)
- `models.py` — the four variants + their Ollama tags. Edit `hf_repo` after a training run.
- `ollama.py` — tiny async/sync wrapper around `/api/chat`, `/api/tags`, `/api/pull`.
- `prompts.py` — shared system prompt. Identical to the one used at training time so we don't drift.

### CADQuery executor (`genesiss.cad`)
- `validator.py` — AST-level deny-list of imports (`os`, `subprocess`, `socket`, …) and builtins (`open`, `eval`, `exec`). Not a sandbox; for real isolation, run the bridge under a container/seccomp profile.
- `executor.py` — exec the script in a fresh dict, expect `result` to be bound, export to STL + STEP via `cadquery.exporters.export`.

### Frontend (`frontend/`)
electron-vite. Main process opens a single window and forwards `GENESISS_BRIDGE_URL` into the renderer via the preload. Renderer is a two-pane React app:
- left: Monaco editor + prompt input + Generate/Run buttons
- right: `@react-three/fiber` canvas loading STL bytes returned by `/exec`.

CSP is locked down to `'self'` plus localhost (the bridge port range).
