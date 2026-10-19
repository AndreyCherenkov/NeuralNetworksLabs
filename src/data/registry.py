from typing import Callable

from torch.utils.data import Dataset

DatasetFactory = Callable[..., Dataset]

DATASET_REGISTRY: dict[str, DatasetFactory] = {}

def register_dataset(name: str, dataset_factory: DatasetFactory) -> None:
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} is already registered")
    DATASET_REGISTRY[name] = dataset_factory

def get_dataset_factory(name: str) -> DatasetFactory:
    try :
        return DATASET_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name}. Available datasets are {list(DATASET_REGISTRY.keys())}"
        ) from None ## TODO zachem?