"""HTTP + WS routes the frontend (or TUI) calls."""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from genesiss.cad.executor import ExecResult, run_cadquery
from genesiss.cad.validator import validate
from genesiss.config import Config
from genesiss.llm.models import REGISTRY
from genesiss.llm.ollama import OllamaClient
from genesiss.llm.prompts import build_messages


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    stream: bool = False


class GenerateResponse(BaseModel):
    code: str
    model: str


class ExecRequest(BaseModel):
    code: str
    export_format: str = "stl"  # stl | step | gltf


def register_routes(app: FastAPI, cfg: Config) -> None:
    client = OllamaClient(cfg.ollama_host)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "model": cfg.model, "ollama": cfg.ollama_host}

    @app.get("/models")
    def list_models() -> dict[str, object]:
        return {"variants": [v.__dict__ for v in REGISTRY.values()], "default": cfg.model}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        variant_name = req.model or cfg.model
        variant = REGISTRY[variant_name]
        messages = build_messages(req.prompt)
        text = await client.chat(variant.ollama_tag, messages)
        return GenerateResponse(code=text, model=variant_name)

    @app.post("/exec")
    async def exec_code(req: ExecRequest) -> ExecResult:
        validate(req.code)
        return run_cadquery(req.code, export_format=req.export_format)

    @app.websocket("/ws/generate")
    async def ws_generate(ws: WebSocket) -> None:
        await ws.accept()
        try:
            payload = await ws.receive_json()
            variant_name = payload.get("model") or cfg.model
            variant = REGISTRY[variant_name]
            messages = build_messages(payload["prompt"])
            async for chunk in client.chat_stream(variant.ollama_tag, messages):
                await ws.send_json({"type": "token", "text": chunk})
            await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            return
        except Exception as e:  # noqa: BLE001
            await ws.send_json({"type": "error", "message": str(e)})
            await ws.close()
