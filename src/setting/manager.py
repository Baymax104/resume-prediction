# -*- coding: UTF-8 -*-
from dynaconf import Dynaconf

from setting.models import *


class SettingManager:
    context_root: Path = Path(__file__).parent.parent.parent

    @classmethod
    def get_settings(cls, env: str) -> Settings:
        source: Dynaconf = Dynaconf(
            root_path=cls.context_root,
            settings_files=["settings.*.yml", ".secrets.*"],
            environments=True,
            env=env,
        )
        all_config = source.as_dict()
        settings = {key.lower(): value for key, value in all_config.items()}
        settings["debug"] = (env != "production")
        settings = Settings.model_validate(settings)
        return settings
