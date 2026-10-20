import hashlib
import json
import subprocess
from datetime import datetime, date
from pathlib import Path
from typing import Any

_ROOT_DIR = Path(__file__).parent.parent.parent


def compute_experiment_id(cfg: dict[str, Any]) -> str:
    params = {
        "commit": get_git_commit_hash(),
        "config_hash": hash_resolved_config(cfg),
        "data_fingerprint": get_data_fingerprint(cfg),
        "split_fingerprint": get_split_fingerprint(cfg),
        "seed": _get_seed(cfg),
    }

    return hash_data(params)


def compute_run_id(exp_id: str) -> str:
    return exp_id + datetime.now().strftime("%Y%m%d%H%M%S")


# TODO git diff saving
def get_git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def get_git_dirty(cwd: Path = _ROOT_DIR) -> bool:
    output = subprocess.check_output(
        ["git", "status", "-s", "--untracked-files=no"], cwd=cwd, stderr=subprocess.DEVNULL
    ).decode("utf-8").strip()
    return bool(output)


def hash_resolved_config(cfg: dict[str, Any]) -> str:
    return hash_data(cfg)

def hash_data(data: Any) -> str:
    normalized = _normalize_data(data)

    if isinstance(normalized, bytes):
        data_bytes = normalized
    else:
        data_bytes = repr(normalized).encode('utf-8')

    return hashlib.sha256(data_bytes).hexdigest()



def get_split_fingerprint(cfg: dict[str, Any]) -> str:
    split_manifest_path = cfg["data"]["split"]["manifest"]

    with open(_ROOT_DIR / split_manifest_path, encoding="utf-8") as f:
        split_data = json.load(f)

    if not isinstance(split_data, dict) or "indices" not in split_data:
        raise ValueError(
            f"Invalid split manifest format by path: {split_manifest_path}"
        )

    indices = split_data["indices"]

    split_indices = {
        name: indices[name]
        for name in ("train", "val", "cal", "test")
    }

    return hash_data(split_indices)

##TODO read
def get_data_fingerprint(cfg: dict[str, Any]) -> str:
    root = _ROOT_DIR / cfg["data"]["root"] / cfg["data"]["name"]
    h = hashlib.sha256()

    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Нет директории сплита: {split_dir}")

        files = sorted(
            (p for p in split_dir.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(root).as_posix(),
        )
        for f in files:
            rel = f.relative_to(root).as_posix().encode("utf-8")
            h.update(len(rel).to_bytes(8, "big"))
            h.update(rel)
            h.update(f.stat().st_size.to_bytes(8, "big"))
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)

    return h.hexdigest()

def _normalize_data(data: Any) -> Any:
    """
    Рекурсивно приводит любые данные к стабильному,
    упорядоченному и сериализуемому виду.
    """
    # 1. Если это словарь, сортируем его по ключам и нормализуем значения
    if isinstance(data, dict):
        return {str(k): _normalize_data(v) for k, v in sorted(data.items())}

    # 2. Если это множество (set), превращаем в отсортированный кортеж
    elif isinstance(data, (set, frozenset)):
        return tuple(_normalize_data(x) for x in sorted(data))

    # 3. Если это списки или кортежи, нормализуем их элементы
    elif isinstance(data, (list, tuple)):
        return tuple(_normalize_data(x) for x in data)

    # 4. Обработка дат и времени
    elif isinstance(data, (datetime, date)):
        return data.isoformat()

    # 5. Если у объекта есть __dict__ (кастомные классы, dataclasses)
    elif hasattr(data, "__dict__"):
        return _normalize_data(data.__dict__)

    # 6. Базовые типы (str, int, float, bool, None, bytes) оставляем как есть
    return data



def _get_seed(cfg: dict[str, Any]) -> str:
    return cfg["experiment"]["seed"]

