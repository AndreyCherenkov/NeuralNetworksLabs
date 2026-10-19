from typing import Any, List

import numpy as np


def get_split_data_indexes(cfg: dict[str, Any], labels: list[Any]) -> dict[str, List[int]]:
    data_split = cfg["data"]["split"]
    rng = np.random.default_rng(seed=data_split["seed"])
    labels = np.asarray(labels)
    classes = np.unique(labels)
    out = {"val": [], "train": [], "cal": []}

    for c in classes:
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = data_split["val_size"] // len(classes)
        n_cal = data_split["cal_size"] // len(classes)
        out["val"] += idx[:n_val].tolist()
        out["cal"] += idx[n_val:n_val + n_cal].tolist()
        out["train"] += idx[n_val + n_cal:].tolist()
    return {k: sorted(v) for k, v in out.items()}
