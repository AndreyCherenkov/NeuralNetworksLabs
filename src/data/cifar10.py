from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.transforms import v2


def create_cifar10(out_path: Path, is_train: bool) -> Dataset:
    transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    return datasets.CIFAR10(
        root=str(out_path),
        train=is_train,
        download=True,
        transform=transform,
    )