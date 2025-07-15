# -*- coding: UTF-8 -*-
import json
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset

from setting import SettingManager


class PersonDataset(Dataset):

    def __init__(self, split: Literal["train", "test"]):
        self.split = split
        settings = SettingManager.get_settings()
        if self.split == "train":
            data_path = settings.data.train
        elif self.split == "test":
            data_path = settings.data.test
        else:
            raise NotImplementedError
        if not data_path:
            raise ValueError(f"Dataset path is empty")
        data_path = Path(data_path)
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.__validate(data)
        self.data = data

    def __validate(self, data: Any):
        if not isinstance(data, list):
            raise ValueError(f"Data must be a list")
        if not all(isinstance(d, dict) for d in data):
            raise ValueError(f"Data item must be a dict")
        if not all("resumes" in d for d in data):
            raise ValueError(f"Data item must have 'resumes' key")

    def __getitem__(self, idx):
        item = self.data[idx]
        pass

    def __len__(self):
        return len(self.data)
