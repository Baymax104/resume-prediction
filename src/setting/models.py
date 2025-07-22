# -*- coding: UTF-8 -*-
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class BaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)


# noinspection PyNestedDecorators
class DataSettings(BaseSettings):
    train: str | None = None
    test: str | None = None
    batch_size: int = 32
    num_workers: int = 4

    @field_validator("train")
    @classmethod
    def trainer_validator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / value
        return str(path.resolve())

    @field_validator("test")
    @classmethod
    def tester_validator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / value
        return str(path.resolve())


# noinspection PyNestedDecorators
class ModelSettings(BaseSettings):
    max_length: int = 32
    window_size: int = 2
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1


# noinspection PyNestedDecorators
class LogSettings(BaseSettings):
    log_dir: str | None = None
    model_dir: str | None = None

    @field_validator("log_dir")
    @classmethod
    def dir_validator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / value
        return str(path.resolve())

    @field_validator("model_dir")
    @classmethod
    def model_dir_validator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / value
        return str(path.resolve())


class TrainSettings(BaseSettings):
    device: str = "cuda"
    epochs: int = 100
    seed: int = 200
    lr: float = 1e-5
    weight_decay: float = 1e-5


class Settings(BaseSettings):
    debug: bool = False
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    log: LogSettings = LogSettings()
    train: TrainSettings = TrainSettings()
