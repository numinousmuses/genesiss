"""FastAPI bridge between the Electron renderer (or TUI) and the local model."""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from genesiss.config import Config
from genesiss.server.routes import register_routes


def build_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="genesiss-bridge", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routes(app, cfg)
    return app


def run(cfg: Config) -> None:
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
