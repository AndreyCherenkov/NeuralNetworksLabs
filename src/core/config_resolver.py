from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime

import yaml

_UNKNOWN_PLACEHOLDER = "??"


def _read_yaml(path: Path, encoding: str = "utf-8") -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding=encoding))


def _deep_merge(
        base: Mapping[str, Any],
        override: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key not in result:
            continue
        current = result[key]
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _profile_extra(
        base: dict[str, Any],
        override: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(base)
    result["experiment"]["seed"] = override["profile"]["base_seed"]
    return result


def _resolve_config(
        base_config: dict[str, Any],
        profile: dict[str, Any],
        mode: dict[str, Any],
) -> dict[str, Any]:
    base_config = _deep_merge(base_config, profile)
    base_config = _deep_merge(base_config, mode)
    base_config = _profile_extra(base_config, profile)

    return base_config


def _save_config(resolved_config: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, allow_unicode=True, sort_keys=False)


def resolve_config(
        base_config: Path,
        profile_path: Path | None = None,
        mode_path: Path | None = None,
        out_path: Path | None = None,
) -> dict[str, Any]:
    root_dir = Path(__file__).resolve().parents[2]

    if profile_path is None:
        profiles_dir = root_dir / "configs/profiles"
        profiles = sorted(profiles_dir.glob("*.yml"))
        if not profiles:
            raise FileNotFoundError(f"No profiles found in {profiles_dir}")
        profile_path = profiles[0]

    if mode_path is None:
        mode_dir = root_dir / "configs/modes"
        profiles = sorted(mode_dir.glob("*.yml"))
        if not profiles:
            raise FileNotFoundError(f"No modes found in {mode_dir}")
        mode_path = mode_dir / profiles[0]

    base_config = _read_yaml(base_config)
    profile = _read_yaml(profile_path)
    mode = _read_yaml(mode_path)

    resolved = _resolve_config(base_config, profile, mode)

    profile_id = profile["profile"]["id"]
    device = resolved["device"]["name"]
    stamp = datetime.now().strftime("%y.%m.%d")

    if out_path is None:
        out_path = root_dir / f"runs/lab0/{profile_id}_{device}_{stamp}/resolved.config.yml"

    _save_config(resolved, out_path)

    return resolved
