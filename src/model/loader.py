from typing import Any

import torch

from model.registry import get_model_factory


def init_model(config: dict[str, Any]) -> torch.nn.Module:
    name = config["model"]["name"]
    factory = get_model_factory(name)
    return factory()
