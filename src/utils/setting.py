# -*- coding: UTF-8 -*-
from pathlib import Path

from pydantic import AfterValidator


def fix_env(env: str) -> str:
    if env in ["dev", "development"]:
        env = "development"
    elif env in ["prod", "production"]:
        env = "production"
    else:
        env = "development"
    return env


def absolute_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (Path(__file__).parent.parent.parent / path).resolve()


AbsolutePath = AfterValidator(absolute_path)
