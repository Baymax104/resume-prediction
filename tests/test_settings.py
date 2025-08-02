# -*- coding: UTF-8 -*-
from setting import SettingManager


def test_dev():
    settings = SettingManager.get_settings("dev")
    assert settings.debug is True
    print(settings)


def test_prod():
    settings = SettingManager.get_settings("prod")
    assert settings.debug is False
