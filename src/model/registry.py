from typing import Callable

import torch

ModelFactory = Callable[..., torch.nn.Module]

MODEL_REGISTRY: dict[str, ModelFactory] = {}

def register_model(name: str, factory: ModelFactory) -> None:
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model {name} is already registered")

    MODEL_REGISTRY[name] = factory

def get_model_factory(name: str) -> ModelFactory:
    try :
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model {name}. Available models are {list(MODEL_REGISTRY.keys())}"
        ) from None