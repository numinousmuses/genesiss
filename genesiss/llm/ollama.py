"""Ollama client wrapper using the official `ollama` Python library.

Why a thin wrapper at all (instead of just using `ollama` directly everywhere)?
    - We want both sync (CLI listing / pulling) and async (FastAPI routes,
      WebSocket streaming) without scattering Client / AsyncClient imports.
    - We can swap implementations later (e.g. for tests) without touching every
      route.

Tool calling / JSON mode? Not in this wrapper. See genesiss/llm/prompts.py for
the rationale — we finetune the model to emit raw CADQuery, so plain chat
completion is the right primitive here. If we ever add structured editing
(parametric tweaks via a tool schema), `ollama-python` supports `tools=` and
`format="json"` natively and we'd extend this wrapper at that point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Iterator

from ollama import AsyncClient, Client


@dataclass
class OllamaModel:
    name: str
    size: int
    digest: str


class OllamaClient:
    def __init__(self, host: str, timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self._sync = Client(host=self.host, timeout=timeout)
        self._async = AsyncClient(host=self.host, timeout=timeout)

    # ---- listing -----------------------------------------------------------
    def list_sync(self) -> list[OllamaModel]:
        resp = self._sync.list()
        out: list[OllamaModel] = []
        for m in resp.get("models", []):
            out.append(
                OllamaModel(
                    name=m.get("name") or m.get("model", ""),
                    size=m.get("size", 0),
                    digest=m.get("digest", ""),
                )
            )
        return out

    # ---- chat --------------------------------------------------------------
    async def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        resp = await self._async.chat(model=model, messages=messages, stream=False)
        return resp["message"]["content"]

    async def chat_stream(
        self, model: str, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        async for chunk in await self._async.chat(model=model, messages=messages, stream=True):
            if chunk.get("done"):
                return
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    # ---- pull (used by `genesiss models pull`) -----------------------------
    def pull_sync(self, name: str) -> Iterator[dict]:
        for evt in self._sync.pull(name, stream=True):
            yield dict(evt)
