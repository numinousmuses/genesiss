"""User config persisted at ~/.config/genesiss/config.toml.

Resolution order: CLI flag > env var > config file > built-in default.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

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

    def with_env(self) -> "Config":
        return replace(
            self,
            host=os.environ.get("GENESISS_HOST", self.host),
            port=int(os.environ.get("GENESISS_PORT", self.port)),
            ollama_host=os.environ.get("OLLAMA_HOST", self.ollama_host),
            model=os.environ.get("GENESISS_MODEL", self.model),
        )


def load() -> Config:
    path = config_path()
    if not path.exists():
        return Config().with_env()
    raw = tomllib.loads(path.read_text())
    return Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__}).with_env()


def save(cfg: Config) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(asdict(cfg)).encode())
    return path


def set_value(key: str, value: str) -> Config:
    """Update a single field by string name. Used by `genesiss config set`."""
    cfg = load()
    if key not in Config.__dataclass_fields__:
        raise KeyError(f"Unknown config key: {key}. Valid: {list(Config.__dataclass_fields__)}")
    field_type = Config.__dataclass_fields__[key].type
    coerced: object = value
    if field_type is int or field_type == "int":
        coerced = int(value)
    new = replace(cfg, **{key: coerced})
    save(new)
    return new
