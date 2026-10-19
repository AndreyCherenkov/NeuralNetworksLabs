from data.cifar10 import create_cifar10
from data.registry import register_dataset

register_dataset("cifar10", create_cifar10)