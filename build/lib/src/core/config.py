import argparse
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

def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        resolved_config = yaml.safe_load(f)
        return resolved_config


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Generate a resolved config file")
    parser.add_argument("--base_config", type=Path, default=root / "configs/lab0.yml")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--mode", type=Path, default=root / "configs/modes/a.yml")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    profile_path = args.profile
    if profile_path is None:
        profiles_dir = root / "configs/profiles"
        profiles = sorted(profiles_dir.glob("*.yml"))
        if not profiles:
            parser.error(f"No profiles found in {profiles_dir}")
        profile_path = profiles[0]

    base_config = _read_yaml(args.base_config)
    profile = _read_yaml(profile_path)
    mode = _read_yaml(args.mode)

    resolved = _load_config(base_config, profile, mode)

    profile_id = profile["profile"]["id"]
    device = resolved["device"]["name"]
    stamp = datetime.now().strftime("%y.%m.%d")
    out_path = args.out or root / f"runs/lab0/{profile_id}_{device}_{stamp}/resolved.config.yml"

    _save_config(resolved, out_path)
