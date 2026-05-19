"""Thin async client over Ollama's HTTP API.

We only use endpoints we actually need:
    GET  /api/tags        list local models
    POST /api/chat        chat completion (with optional streaming)
    POST /api/pull        pull a model (NDJSON progress stream)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import httpx


@dataclass
class OllamaModel:
    name: str
    size: int
    digest: str


class OllamaClient:
    def __init__(self, host: str, timeout: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ---- listing -----------------------------------------------------------
    def list_sync(self) -> list[OllamaModel]:
        r = httpx.get(f"{self.host}/api/tags", timeout=self.timeout)
        r.raise_for_status()
        return [
            OllamaModel(name=m["name"], size=m.get("size", 0), digest=m.get("digest", ""))
            for m in r.json().get("models", [])
        ]

    # ---- chat --------------------------------------------------------------
    async def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(
                f"{self.host}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    async def chat_stream(
        self, model: str, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            async with c.stream(
                "POST",
                f"{self.host}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("done"):
                        return
                    chunk = obj.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

    # ---- pull (CLI helper) -------------------------------------------------
    def pull_sync(self, name: str) -> Iterator[dict]:
        with httpx.stream(
            "POST", f"{self.host}/api/pull", json={"name": name}, timeout=None
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)
