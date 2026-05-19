"""Cheap static checks before we exec model output.

This is NOT a sandbox — CADQuery scripts call into native libraries by design.
We only reject the obviously hostile (os/sys/subprocess imports, file writes
outside the scratch dir, network access). For real isolation, run the executor
behind a container or seccomp profile.
"""
from __future__ import annotations

import ast

BANNED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "httpx",
    "pathlib",
    "ctypes",
    "multiprocessing",
}
BANNED_BUILTINS = {"open", "exec", "eval", "compile", "__import__"}


class UnsafeCode(ValueError):
    pass


def validate(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] in BANNED_MODULES:
                    raise UnsafeCode(f"banned import: {n.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BANNED_MODULES:
                raise UnsafeCode(f"banned import from: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_BUILTINS:
                raise UnsafeCode(f"banned call: {node.func.id}")
