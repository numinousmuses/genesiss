"""User config persisted at ~/.config/genesiss/config.toml.

Resolution order: CLI flag > env var > config file > built-in default.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import tomli_w
from platformdirs import user_config_dir

Mode = Literal["gui", "headless"]
DEFAULT_VARIANT = "genesiss-4b"


def config_path() -> Path:
    return Path(user_config_dir("genesiss")) / "config.toml"


@dataclass(frozen=True)
class Config:
    default_mode: Mode = "gui"
    model: str = DEFAULT_VARIANT
    host: str = "127.0.0.1"
    port: int = 8765
    ollama_host: str = "http://127.0.0.1:11434"
    # Absolute path to the Electron frontend dir. Set by the user when they've
    # installed genesiss as a tool but kept a clone of the repo around for the
    # GUI. None means "look next to this package" (dev-mode default).
    frontend_dir: Optional[str] = None

    def with_env(self) -> "Config":
        return replace(
            self,
            host=os.environ.get("GENESISS_HOST", self.host),
            port=int(os.environ.get("GENESISS_PORT", self.port)),
            ollama_host=os.environ.get("OLLAMA_HOST", self.ollama_host),
            model=os.environ.get("GENESISS_MODEL", self.model),
            frontend_dir=os.environ.get("GENESISS_FRONTEND_DIR", self.frontend_dir),
        )


def _normalize_key(key: str) -> str:
    """Accept both `default-mode` and `default_mode` on the CLI."""
    return key.replace("-", "_")


def load() -> Config:
    path = config_path()
    if not path.exists():
        return Config().with_env()
    raw = tomllib.loads(path.read_text())
    return Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__}).with_env()


def save(cfg: Config) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # tomli_w can't serialize None; drop unset optional fields.
    payload = {k: v for k, v in asdict(cfg).items() if v is not None}
    path.write_bytes(tomli_w.dumps(payload).encode())
    return path


def set_value(key: str, value: str) -> Config:
    """Update a single field by string name. Used by `genesiss config set`.

    Accepts both kebab-case (`default-mode`) and snake_case (`default_mode`).
    """
    key = _normalize_key(key)
    cfg = load()
    if key not in Config.__dataclass_fields__:
        valid = sorted(Config.__dataclass_fields__)
        raise KeyError(f"Unknown config key: {key}. Valid: {valid}")
    field_type = Config.__dataclass_fields__[key].type
    coerced: object = value
    if field_type is int or field_type == "int":
        coerced = int(value)
    elif key == "frontend_dir":
        # Resolve to an absolute path so `cd`-ing later doesn't break it.
        coerced = str(Path(value).expanduser().resolve())
    new = replace(cfg, **{key: coerced})
    save(new)
    return new
