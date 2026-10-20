import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

Rule = dict[str, Any]
KindFunction = Callable[[str, Any], Any]
_KINDS: dict[str, KindFunction] = {}
F = TypeVar("F", bound=KindFunction)


def kind(name: str) -> Callable[[F], F]:
    def register(func: F) -> F:
        if name in _KINDS:
            raise ValueError(f"Kind {name!r} already registered")
        _KINDS[name] = func
        return func

    return register


def _slice(h: str, rule: Rule) -> str:
    a, b = rule["slice"]
    return h[a:b]


@kind("prefix")
def _prefix(h: str, rule: Rule) -> str:
    return rule["prefix"] + _slice(h, rule)


@kind("hash")
def _hash(h: str, rule: Rule) -> str:
    return h


@kind("mod_number")
def _mod(h: str, rule: Rule) -> int:
    return int(_slice(h, rule), 16) % rule["mod"]


@kind("choice")
def _choice(h: str, rule: Rule) -> Any:
    choices = rule["choices"]
    return choices[int(_slice(h, rule), 16) % len(choices)]


def _load_specs(spec_path: Path) -> dict:
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    fields = set()
    for rule in data["fields"]:
        if rule["kind"] not in _KINDS:
            raise ValueError(f"Unknown kind {rule['kind']}")
        if rule["field"] in fields:
            raise ValueError(f"Duplicate field {rule['field']}")
        fields.add(rule["field"])
    return data


def _generate_hex_hash(profile_id: str, encoding='utf-8') -> str:
    return hashlib.sha256(profile_id.encode(encoding=encoding)).hexdigest()


# TODO delete?
def profile_fields(specification: dict) -> frozenset[str]:
    return frozenset(rule["field"] for rule in specification["fields"])


def generate_profile(h: str, out_path: Path, specification_path: Path) -> None:
    specification = _load_specs(specification_path)
    data: dict[str, dict[str, Any]] = dict()
    for rule in specification["fields"]:
        section, field = rule["field"].split(".")
        if section not in data:
            data[section] = dict()
        data[section][field] = _KINDS[rule["kind"]](h, rule)

    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description='Generate a profile file')
    parser.add_argument('--id', required=True, type=str)
    parser.add_argument('--out', required=False, type=str, default=None)
    parser.add_argument('--specs', required=False, type=str, default=None)

    args = parser.parse_args()
    hex_hash = _generate_hex_hash(args.id)
    specs_path = args.specs or f'./configs/profile_specs.yml'
    out_path = args.out or f'./configs/profiles/{hex_hash}.yml'
    generate_profile(hex_hash, Path(out_path), Path(specs_path))

if __name__ == '__main__':
    main()
