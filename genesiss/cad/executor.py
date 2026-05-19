"""Execute a CADQuery script and export the result for the viewer.

The script must assign its final shape to a top-level name `result`. We then
export it to STL (for three.js) and STEP (for download). GLTF is left for a
future pass once we wire up an OCC→glTF converter.
"""
from __future__ import annotations

import io
import tempfile
import traceback
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ExportFormat = Literal["stl", "step"]


class ExecResult(BaseModel):
    ok: bool
    error: str | None = None
    traceback: str | None = None
    stl_b64: str | None = None
    step_b64: str | None = None
    log: str = ""


def _to_b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def run_cadquery(code: str, export_format: ExportFormat = "stl") -> ExecResult:
    """Exec `code` in an isolated namespace. The result must be bound to `result`.

    We don't try to sandbox here — see cad.validator for the AST-level checks.
    Run this process under OS-level isolation if you don't trust inputs.
    """
    import cadquery as cq

    buf = io.StringIO()
    ns: dict[str, object] = {"cq": cq, "cadquery": cq}
    try:
        exec(compile(code, "<genesiss>", "exec"), ns, ns)  # noqa: S102 — intentional
    except Exception as e:  # noqa: BLE001
        return ExecResult(ok=False, error=str(e), traceback=traceback.format_exc(), log=buf.getvalue())

    shape = ns.get("result")
    if shape is None:
        return ExecResult(
            ok=False,
            error="script did not bind `result`",
            log=buf.getvalue(),
        )

    out = ExecResult(ok=True, log=buf.getvalue())
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        if export_format in ("stl",):
            stl_path = tdp / "out.stl"
            cq.exporters.export(shape, str(stl_path))
            out.stl_b64 = _to_b64(stl_path.read_bytes())
        step_path = tdp / "out.step"
        cq.exporters.export(shape, str(step_path))
        out.step_b64 = _to_b64(step_path.read_bytes())
    return out
