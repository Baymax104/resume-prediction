# -*- coding: UTF-8 -*-
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from setting import SettingManager


class ResumeDataset(Dataset):

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
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_length = settings.data.max_length

    def __validate(self, data: Any):
        if not isinstance(data, list):
            raise ValueError(f"Data must be a list")
        if not all(isinstance(d, dict) for d in data):
            raise ValueError(f"Data item must be a dict")
        if not all("resumes" in d for d in data):
            raise ValueError(f"Data item must have 'resumes' key")

    def __getitem__(self, idx):
        item = self.data[idx]
        resumes = item["resumes"]

        # split time and position
        times = []
        positions = []
        for resume in resumes:
            time, position = resume.split(" ", 1)
            times.append(time.strip())
            positions.append(position.strip())

        # extract time features
        time_features = []
        for time in times:
            start, end = time.split("-", 1)
            start_year, start_month = map(int, re.findall(r"\d+", start))
            start_time = datetime(start_year, start_month, 1)

            if end.strip() in ["", "至今"]:
                end_year, end_month = datetime.now().year, datetime.now().month  # current time as end time
            else:
                end_year, end_month = map(int, re.findall(r"\d+", end))
            end_time = datetime(end_year, end_month, 1)

            # 计算月数差作为特征
            months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
            time_features.append([start_year, start_month, end_year, end_month, months])

        position_features = []
        for position in positions:
            encoded = self.tokenizer.encode_plus(
                position,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            position_features.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze()
            })

        samples = []
        for i in range(len(times) - 1):
            input_times = torch.tensor(time_features[:i + 1])
            input_positions = position_features[:i + 1]
            target_time = torch.tensor(time_features[i + 1])
            target_position = position_features[i + 1]
            samples.append({
                "input_times": input_times,
                "input_positions": input_positions,
                "target_time": target_time,
                "target_position": target_position,
            })

        return samples

    def __len__(self):
        return len(self.data)
