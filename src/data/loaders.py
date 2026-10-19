from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from data.registry import get_dataset_factory

_DATASETS = (
    "cifar10",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _make_dataset_dir(config: dict[str, Any], path: Path = _PROJECT_ROOT) -> Path:
    data_config = config["data"]
    dataset_name = data_config["name"]
    dataset_path = (
            path
            / data_config.get("root", "data")
            / dataset_name
    )

    if dataset_name not in _DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported")

    dataset_path.mkdir(parents=True, exist_ok=True)
    return dataset_path


def _load_dataset(config: dict[str, Any], is_train: bool) -> Any:
    name = config["data"]["name"]
    root = _make_dataset_dir(config) / ("train" if is_train else "test")
    print(f"Loading dataset '{name}' from '{root}'")
    factory = get_dataset_factory(name)
    return factory(root, is_train)


# TODO workers
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

##TODO batch size, shuffle, num_workers, worker_init_fn. Example below
# def make_loader(dataset, batch_size, seed, num_workers, shuffle=True):
#     g = torch.Generator()
#     g.manual_seed(seed)
#     return torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=shuffle,
#         num_workers=num_workers, worker_init_fn=seed_worker, generator=g,
#     )

def _init_generator(config: dict[str, Any]) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(config["experiment"]["seed"])  # TODO refactor
    return gen


def make_train_loader(config: dict[str, Any]) -> Any:
    gen = _init_generator(config)
    train_fraction = config["data"]["train_fraction"]

    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"Train fraction must be between in (0.0, 1.0]. Actual: {train_fraction}")

    train = _load_dataset(config, is_train=True)

    train_size = int(train_fraction * len(train))
    indices = torch.randperm(len(train), generator=gen)[:train_size].tolist()
    train_subset = Subset(train, indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        generator=gen
    )

    return train_loader


def make_test_loader(config: dict[str, Any]) -> Any:
    gen = _init_generator(config)
    test = _load_dataset(config, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test,
        generator=gen
    )
    return test_loader
