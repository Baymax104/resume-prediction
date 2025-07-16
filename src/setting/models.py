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
    max_length: int = 32

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
class LogSettings(BaseSettings):
    dir: str | None = None

    @field_validator("dir")
    @classmethod
    def dir_validator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / value
        return str(path.resolve())


class TrainSettings(BaseSettings):
    lr: float = 1e-5


class Settings(BaseSettings):
    data: DataSettings = DataSettings()
    log: LogSettings = LogSettings()
    train: TrainSettings = TrainSettings()
