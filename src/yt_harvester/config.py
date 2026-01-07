"""Configuration loading for `yt_harvester`.

This module keeps configuration intentionally simple:
- Load `config.yaml` (or user-provided path) when present
- Deep-merge nested dictionaries into defaults
- Let CLI flags override configuration values
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml

DEFAULT_CONFIG = {
    "comments": {
        "top_n": 20,
        "max_download": 10000
    },
    "output": {
        "format": "txt",
        "dir": "."
    },
    "processing": {
        "sentiment": True,
        "keywords": True
    }
}

def _deep_merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge one mapping into another.

    Why: YAML config is naturally nested (e.g. `comments.top_n`). A shallow
    `dict.update()` would replace entire sub-dicts and accidentally drop defaults.

    Args:
        dst: Destination mapping (mutated in place).
        src: Source mapping to merge into `dst`.

    Returns:
        The same `dst` object for convenience.
    """

    for key, value in src.items():
        if (
            key in dst
            and isinstance(dst[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(dst[key], value)  # type: ignore[arg-type]
        else:
            dst[key] = value
    return dst


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration and deep-merge into defaults.

    Args:
        config_path: Path to a YAML file (default: `config.yaml`).

    Returns:
        A config dictionary. If the file is missing or invalid, returns defaults.
    """

    path = Path(config_path)
    config: Dict[str, Any] = {
        # Make a deep-ish copy so DEFAULT_CONFIG isn't mutated by merges.
        "comments": dict(DEFAULT_CONFIG["comments"]),
        "output": dict(DEFAULT_CONFIG["output"]),
        "processing": dict(DEFAULT_CONFIG["processing"]),
    }

    if not path.exists():
        return config

    try:
        with path.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
    except Exception:
        # Keep stdout clean; callers can decide how to surface this.
        return config

    if isinstance(user_config, dict) and user_config:
        _deep_merge(config, user_config)

    return config
