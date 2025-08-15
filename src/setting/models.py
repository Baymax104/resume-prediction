# -*- coding: UTF-8 -*-
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from utils import AbsolutePath


class BaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True, strict=False)


class DataSettings(BaseSettings):
    train: Annotated[Path | None, AbsolutePath] = None
    test: Annotated[Path | None, AbsolutePath] = None
    batch_size: int = 32
    num_workers: int = 4


class ModelSettings(BaseSettings):
    pretrained_model_dir: Annotated[Path | None, AbsolutePath] = None
    max_length: int = 32
    window_size: int = 2
    d_model: int = 512
    num_layers: int = 6
    dropout: float = 0.1


class LogSettings(BaseSettings):
    log_dir: Annotated[Path | None, AbsolutePath] = None


class TrainSettings(BaseSettings):
    device: str = "cuda"
    epochs: int = 100
    seed: int = 200
    lr: float = 1e-5
    weight_decay: float = 1e-5
    checkpoint_dir: Annotated[Path | None, AbsolutePath] = None
    checkpoint_step: int = 50


class Settings(BaseSettings):
    debug: bool = False
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    log: LogSettings = LogSettings()
    train: TrainSettings = TrainSettings()
