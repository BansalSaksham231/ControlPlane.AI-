"""
Centralised access to ``config/settings.yaml``.

Every engine and detector in ControlPlane.ai reads its thresholds from
this single configuration file rather than hard-coding magic numbers.
The loader caches parsed configuration by absolute path so repeated
calls in a request path are cheap.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

# Repository root (this file lives at the root).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "settings.yaml")


@lru_cache(maxsize=8)
def _load_cached(abs_path: str) -> dict[str, Any]:
    with open(abs_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {abs_path} did not parse to a mapping")
    return data


def load_settings(path: str | None = None) -> dict[str, Any]:
    """
    Load and return the ControlPlane configuration as a plain dict.

    ``path`` may be relative (resolved against the repo root) or absolute.
    Results are cached; pass ``clear_settings_cache()`` between tests that
    mutate the file on disk.
    """
    if path is None:
        abs_path = DEFAULT_CONFIG_PATH
    elif os.path.isabs(path):
        abs_path = path
    else:
        abs_path = os.path.join(REPO_ROOT, path)
    # Return a shallow copy so callers cannot mutate the cached object.
    return dict(_load_cached(abs_path))


def clear_settings_cache() -> None:
    """Drop the parsed-configuration cache (useful in tests)."""
    _load_cached.cache_clear()
