"""Shared filesystem paths."""
from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir


def cache_dir() -> Path:
    d = Path(user_cache_dir("genesiss"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    d = Path(user_data_dir("genesiss"))
    d.mkdir(parents=True, exist_ok=True)
    return d
