"""Salvage Python source from a model's raw chat output.

Even a well-finetuned model will occasionally:
    - wrap output in ```python ... ``` fences,
    - emit a one-line preamble ("Sure, here's the code:") before the body,
    - end with a sign-off ("Let me know if you want to tweak it.").

This module collapses those cases down to a single best-effort Python string.
It is *not* a security boundary — see genesiss.cad.validator for the AST
deny-list that runs before exec.
"""
from __future__ import annotations

import ast
import re

# Triple-fence block, optional language tag.
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the most likely Python body from `text`.

    Strategy (in order):
      1. If one or more ``` fenced blocks exist, return the *longest* one.
         (Models sometimes emit multiple, with the "real" one being the longest.)
      2. Otherwise, strip leading "Here's …:" / "Sure, …" preambles and return
         the result.
    """
    text = text.strip()

    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()

    # No fences. Try to peel off a chatty first line if the rest looks like code.
    lines = text.splitlines()
    if len(lines) >= 2 and _looks_chatty(lines[0]) and _looks_codey("\n".join(lines[1:])):
        return "\n".join(lines[1:]).strip()
    return text


def _looks_chatty(line: str) -> bool:
    s = line.strip().lower()
    return bool(s) and not s.startswith(("import ", "from ", "#", "def ", "class ", "result", "cq.", "cadquery"))


def _looks_codey(text: str) -> bool:
    head = text.lstrip().splitlines()[0] if text.strip() else ""
    return head.startswith(("import ", "from ", "#", "def ", "class ", "result", "cq.", "cadquery"))


def parses(code: str) -> tuple[bool, str | None]:
    """Return (True, None) if `code` is syntactically valid Python, else (False, msg)."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"
    return True, None
