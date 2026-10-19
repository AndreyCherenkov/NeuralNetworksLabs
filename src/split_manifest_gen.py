import argparse
import json
from pathlib import Path

import yaml

from data.loaders import load_dataset
from data.split import get_split_data_indexes

_ROOT_DIR = Path(__file__).resolve().parents[1]


def _get_labels(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)

    raise AttributeError(
        f"{type(dataset).__name__} has no targets or labels attribute"
    )


def _resolve_split_manifest(base_cfg_path: Path) -> None:
    base_cfg = yaml.safe_load(base_cfg_path.read_text())
    labels = _get_labels(load_dataset(base_cfg, True))

    split_data = base_cfg["data"]["split"]
    roles = get_split_data_indexes(base_cfg, labels)
    roles["test"] = list(range(split_data["test_size"]))

    manifest = {
        "dataset": base_cfg["data"]["name"],
        "version": split_data["version"],
        "split_seed": split_data["seed"],
        "sizes": {
            "train": split_data["train_size"],
            "val": split_data["val_size"],
            "cal": split_data["cal_size"],
            "test": split_data["test_size"],
        },
        "indices": roles,
    }

    out_path = Path(_ROOT_DIR / split_data["manifest"])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str, required=False)
    args = parser.parse_args()

    base_cfg_path = (
        Path(args.base_cfg)
        if args.base_cfg
        else _ROOT_DIR / "configs" / "lab0.yml"
    )

    _resolve_split_manifest(base_cfg_path)


if __name__ == "__main__":
    main()
