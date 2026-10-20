import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from core.identity import compute_experiment_id, compute_run_id, get_git_commit_hash, get_git_dirty, \
    hash_resolved_config, get_data_fingerprint, get_split_fingerprint
from model.loader import init_model

_ROOT_DIR = Path(__file__).parent.parent.parent

def resolve_manifest(cfg_path: Path, out: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    exp_id = compute_experiment_id(cfg)

    dt = datetime.now().astimezone()

    cudnn_version = (
        str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else "unavailable"
    )

    device_type = cfg["device"]["name"]
    device_name = (
        _get_cpu_info()
        if device_type.lower() == "cpu"
        else "unknown"
    )

    model = init_model(cfg)
    model_params_number = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params_number = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad  # TODO Zachem
    )

    manifest: dict[str, Any] = {
        "exp_id": exp_id,
        "run_id": compute_run_id(exp_id),
        "profile": {
            "id": cfg["profile"]["id"],
            "id_hash": cfg["profile"]["id_hash"],
        },
        "date": dt.strftime("%Y%m%d"),
        "time": dt.strftime("%H%M%S"),
        "git": {
            "commit_hash": get_git_commit_hash(),
            "dirty": get_git_dirty(),
        },
        "config": {
            "path": str(cfg_path.resolve()),
            "hash": hash_resolved_config(cfg),
        },
        "data": {
            "fingerprint": get_data_fingerprint(cfg),
            "split_fingerprint": get_split_fingerprint(cfg),
        },
        "experiment": {
            "seed": cfg["experiment"]["seed"],
            "deterministic": cfg["experiment"]["deterministic"],
            # TODO amp, num_workers
            # "amp": amp,
            # "num_workers": num_workers,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "pytorch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            # TODO hash файла блокировки зависимостей
        },
        "hardware": {
            "device_type": device_type,
            "device_name": device_name, # TODO дописать для GPU

        },
        "model": {
            "class_name": model.__class__.__name__,
            "total_parameters": model_params_number,
            "trainable_parameters": trainable_params_number,
        },
    }

    # Save manifest.json
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

def load_split_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    split_cfg = Path(__file__).parent.parent.parent / cfg["data"]["split"]["manifest"]
    with split_cfg.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest

def _get_cpu_info() -> str:
    cpu_name = platform.processor() or platform.machine()

    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            cpu_name = result.stdout.strip() or cpu_name
        except (subprocess.SubprocessError, OSError):
            pass

    return cpu_name
