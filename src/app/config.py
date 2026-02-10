"""Application configuration loaded from YAML with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root: two levels up from this file (src/app/config.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"
_CONFIG_EXAMPLE_PATH = _PROJECT_ROOT / "config" / "config.example.yaml"

# Environment variable -> config key path mapping.
# Each entry maps ENV_VAR to a dot-separated path into the config dict.
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "SLACK_WEBHOOK_URL": ("alerts.slack.webhook_url", str),
    "SLACK_ENABLED": ("alerts.slack.enabled", bool),
    "MSL_DB_PATH": ("storage.db_path", str),
    "MSL_PARQUET_DIR": ("storage.parquet_dir", str),
    "MSL_SERVER_HOST": ("server.host", str),
    "MSL_SERVER_PORT": ("server.port", int),
    "MSL_LOG_LEVEL": ("log_level", str),
    "OLLAMA_BASE_URL": ("ai.ollama.base_url", str),
    "OLLAMA_MODEL": ("ai.ollama.model", str),
}

_instance: dict[str, Any] | None = None


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated key path."""
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _cast(value: str, target_type: type) -> Any:
    """Cast a string environment variable to the target type."""
    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    return target_type(value)


def _load_yaml() -> dict[str, Any]:
    """Load YAML config, falling back to the example file."""
    path = _CONFIG_PATH if _CONFIG_PATH.exists() else _CONFIG_EXAMPLE_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """Override config values with environment variables when set."""
    for env_var, (dotted_key, target_type) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            _set_nested(cfg, dotted_key, _cast(value, target_type))


def load_config(*, reload: bool = False) -> dict[str, Any]:
    """Load and return the application config (singleton).

    Args:
        reload: Force a fresh load, bypassing the cached instance.

    Returns:
        The merged configuration dictionary.
    """
    global _instance
    if _instance is not None and not reload:
        return _instance

    cfg = _load_yaml()
    _apply_env_overrides(cfg)
    _instance = cfg
    return _instance


def get_config(section: str | None = None) -> dict[str, Any]:
    """Get the full config or a specific top-level section.

    Args:
        section: Optional top-level key (e.g. "storage", "alerts").
                 Returns the full config dict when None.

    Returns:
        Config dictionary (full or section).

    Raises:
        KeyError: If the requested section does not exist.
    """
    cfg = load_config()
    if section is None:
        return cfg
    if section not in cfg:
        raise KeyError(f"Config section '{section}' not found")
    return cfg[section]
