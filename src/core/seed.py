import random
from typing import Any

import numpy as np
import torch


##TODO rename package
##TODO read docks
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# TODO log GLOBAL_SEED value
def set_seed(config: dict[str, Any], seed_increment: int = 0) -> None:
    experiment = config.get("experiment", {})
    seed = experiment.get("seed")

    if seed is None:
        raise ValueError("Seed not found in config")

    seed += seed_increment

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    deterministic = experiment.get("deterministic", False)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
