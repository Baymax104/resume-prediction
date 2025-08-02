# -*- coding: UTF-8 -*-
import time
from typing import Any, Literal

import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from setting import Settings


class Recorder:

    def __init__(self, metrics: list[str], mode: Literal["train", "test"], settings: Settings):
        self.metrics = metrics
        self.mode = mode
        self.log_dir = settings.log.log_dir
        self.train = self.mode != "test"
        field_names = (["Epoch"] if self.train else []) + [metric.title() for metric in self.metrics]
        self.prettytable = PrettyTable(field_names=field_names)
        self.record_dict = {metric: [] for metric in self.metrics}
        self.step = 0
        self.checkpoints_dir = settings.train.checkpoint_dir

    def add_record(self, step_dict: dict[str, Any]):
        for metric, value in step_dict.items():
            if metric in self.record_dict:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                self.record_dict[metric].append(value)

        step_column = [self.step + 1] if self.train else []
        row = step_column + [self.record_dict[metric][-1] for metric in self.metrics]
        self.prettytable.add_row(row)
        self.step += 1

    def print(self, clean=False):
        print(self.prettytable)
        if clean:
            self.prettytable.clear_rows()

    def plot(self):
        for metric, values in self.record_dict.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(metric)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.plot(values)
            log_path = self.log_dir / f"{metric}-{time.strftime("%Y%m%d-%H%M%S")}.jpg"
            plt.savefig(log_path)

    def keys(self) -> list[str]:
        return list(self.record_dict.keys())

    def __getitem__(self, item):
        if item not in self.record_dict:
            raise KeyError(f"Key {item} is not in {self.metrics}")
        return self.record_dict[item] if self.train else self.record_dict[item][0]

    def __str__(self):
        return str(self.record_dict)

    def save_checkpoint(self, model, optimizer, epoch):
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        states = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{epoch}.pt"
        torch.save(states, checkpoint_path)

    def load_checkpoint(self, checkpoint: int) -> dict:
        if not self.checkpoints_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {self.checkpoints_dir} does not exist")
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{checkpoint}.pt"
        return torch.load(checkpoint_path)
