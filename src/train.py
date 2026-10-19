from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.config_resolver import resolve_config
from core import seed
from data.loaders import make_train_loader
from model.loader import init_model


def _read_config(
        base_config: Path,
        profile_path: Path | None = None,
        mode_path: Path | None = None,
        out_path: Path | None = None
) -> dict[str, Any]:
    return resolve_config(base_config)

def _init_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    name = config["optimizer"]["name"]
    match name:
        case "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config["optimizer"]["lr"],
                weight_decay=config["optimizer"]["weight_decay"],
            )
        case "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config["optimizer"]["lr"],
                weight_decay=config["optimizer"]["weight_decay"],
            )
        case _:
            raise ValueError(f"Unknown optimizer {name}")

def _init_scheduler(config: dict[str, Any]):
    name = config["scheduler"]["name"]
    match name:
        case "step":
            return torch.optim.lr_scheduler.StepLR
        case "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR()
        case _:
            raise ValueError(f"Unknown scheduler {name}")


def _set_state(config: dict[str, Any], seed_increment: int = 0) -> None:
    seed.set_seed(config, seed_increment)
    seed.set_deterministic(config)

##TODO отрефакторить под разные датасеты, добавить опцианальные пути к профилю и режиму
def _train(cfg_path: Path):
    config = resolve_config(cfg_path)

    _set_state(config) ##TODO не сбрасывает ли split состояние?

    device = torch.device(config["device"]["name"])

    model = init_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _init_optimizer(config, model)
    #TODO scheduler

    epochs = config["train"]["epochs"]

    train_loader = make_train_loader(config)

    print(f"Training with {epochs} epochs starting") #TODO в логер
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0 ##TODO for what?

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

        #TODO вынести в логер; LOGER START
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                    output.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2%}"
        )
        #LOGER END

    return model

def main():
    path = Path("../configs/lab0.yml")
    _train(path)

if __name__ == "__main__":
    main()
    #TODO args, scheduler and optimizer registry

