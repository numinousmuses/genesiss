"""Prompt construction for genesiss variants.

All variants are finetuned on CADQuery generation, so the system prompt is
intentionally terse — the model has already learned the format. We give it a
hard rule to emit only python code.
"""
from __future__ import annotations

SYSTEM = (
    "You are Genesiss, a CAD assistant. "
    "Given a user description, output a complete CADQuery Python script that builds the part. "
    "Output ONLY python — no markdown fences, no commentary. "
    "End the script with `result = <final_workplane_or_assembly>` so it can be exported."
)


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
