"""Prompt construction for genesiss variants.

All variants are finetuned on CADQuery generation, so the system prompt is
intentionally terse — the model has already learned the format. We give it a
hard rule to emit only python code.

Why plain chat completion (no tool calling, no JSON mode)?
    - The output is Python source, not a structured object. Wrapping it in
      `{"code": "..."}` or a tool call adds an escaping layer that doesn't
      help (and routinely breaks on long multi-line Python with quotes).
    - Training-inference parity. Our SFT dataset is `{"input", "output"}`
      where `output` is raw CADQuery; the assistant turn the model learns is
      plain code. Switching to JSON/tool format at inference time would
      introduce a distribution shift the model wasn't trained on.
    - Tool calling pays for itself when the model decides *whether* and *how*
      to call something — here there's exactly one thing to do every turn
      (emit a CADQuery script), so the dispatch the tool layer is meant to
      handle just isn't there.
    - If we later add parametric structured editing (e.g. "change the hole
      diameter to 12mm" → tool call with `{type:"hole", d_mm:12}`), that's
      the moment to introduce tools — and we'd train a separate adapter or
      add tool-format rows to the SFT dataset. Don't do it preemptively.
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
