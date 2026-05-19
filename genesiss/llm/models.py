"""Genesiss variant registry.

`hf_repo` is where the finetuned LoRA/merged model lives on the Hub (filled in
after a training run). `ollama_tag` is what `ollama pull` / `ollama run` uses.
Update these as runs complete.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    base: str
    hf_repo: str
    ollama_tag: str
    context: int


REGISTRY: dict[str, Variant] = {
    "genesiss-4b": Variant(
        name="genesiss-4b",
        base="unsloth/Qwen3.5-4B",
        hf_repo="genesiss/genesiss-4b",
        ollama_tag="genesiss-4b:latest",
        context=8192,
    ),
    "genesiss-9b": Variant(
        name="genesiss-9b",
        base="unsloth/Qwen3.5-9B",
        hf_repo="genesiss/genesiss-9b",
        ollama_tag="genesiss-9b:latest",
        context=8192,
    ),
    "genesiss-20b": Variant(
        name="genesiss-20b",
        base="unsloth/gpt-oss-20b",
        hf_repo="genesiss/genesiss-20b",
        ollama_tag="genesiss-20b:latest",
        context=8192,
    ),
    "genesiss-27b": Variant(
        name="genesiss-27b",
        base="unsloth/Qwen3.5-27B",
        hf_repo="genesiss/genesiss-27b",
        ollama_tag="genesiss-27b:latest",
        context=8192,
    ),
}
