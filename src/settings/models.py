# -*- coding: UTF-8 -*-
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class BaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)


# noinspection PyNestedDecorators
class SystemSettings(BaseSettings):
    data_dir: str | None = None
    log_dir: str | None = None

    @field_validator("data_dir")
    @classmethod
    def data_dir_validator(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / value
        return str(path.resolve())

    @field_validator("log_dir")
    @classmethod
    def log_dir_validator(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / value
        return str(path.resolve())


class Settings(BaseSettings):
    system: SystemSettings = SystemSettings()
