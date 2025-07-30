# -*- coding: UTF-8 -*-
from setting import SettingManager


def test_dev():
    settings = SettingManager.get_settings()
    assert settings.debug is True
    print(settings)


def test_prod():
    # os.environ["ENV"] = "production"
    settings = SettingManager.get_settings()
    assert settings.debug is False
