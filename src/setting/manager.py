# -*- coding: UTF-8 -*-
from dynaconf import Dynaconf
from ruamel.yaml import YAML

from setting.models import *


class SettingManager:
    context_root: Path = Path(__file__).parent.parent.parent
    setting_path: Path = context_root / "settings.yml"
    all_config: Dynaconf = Dynaconf(
        root_path=context_root,
        settings_files=[setting_path.name, ".secrets.*"],
    )

    @classmethod
    def get_settings(cls) -> Settings:
        if not cls.setting_path.exists():
            settings = cls.__init_settings()
            with cls.setting_path.open("w", encoding="utf-8") as f:
                YAML().dump(settings.model_dump(), f)
            return settings

        all_config = cls.all_config.as_dict()
        settings = {key.lower(): value for key, value in all_config.items()}
        settings = Settings.model_validate(settings)
        return settings

    @classmethod
    def __init_settings(cls) -> Settings:
        return Settings()


if __name__ == "__main__":
    settings = SettingManager.get_settings()
    print(settings)
