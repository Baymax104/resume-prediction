# -*- coding: UTF-8 -*-
import os

from dynaconf import Dynaconf

from setting.models import *


class SettingManager:
    context_root: Path = Path(__file__).parent.parent.parent
    env = os.getenv("ENV", "development")
    source: Dynaconf = Dynaconf(
        root_path=context_root,
        settings_files=["settings.*.yml", ".secrets.*"],
        environments=True,
        default_env="development",
        env=env,
    )

    @classmethod
    def get_settings(cls) -> Settings:
        all_config = cls.source.as_dict()
        settings = {key.lower(): value for key, value in all_config.items()}
        settings["debug"] = (cls.env != "production")
        settings = Settings.model_validate(settings)
        return settings
