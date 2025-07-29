# -*- coding: UTF-8 -*-
import os


def get_env() -> str:
    env = os.getenv("ENV", "development")
    if env in ["dev", "development"]:
        env = "development"
    elif env in ["prod", "production"]:
        env = "production"
    else:
        env = "development"
    return env
