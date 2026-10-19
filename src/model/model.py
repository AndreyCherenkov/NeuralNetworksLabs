from typing import Any

import torch

from model.small_cnn import SmallCnn


def init_model(config: dict[str, Any]) -> torch.nn.Module:
    device = torch.device(config["device"]["name"])
    experiment_name = config["model"]["name"]

    match experiment_name:
        case "small_cnn":
            return SmallCnn().to(device)
        case _:
            raise ValueError(f"Unknown experiment name: {experiment_name}")
