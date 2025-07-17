# -*- coding: UTF-8 -*-
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from setting import SettingManager, Settings


class ResumeDataset(Dataset):

    def __init__(self, split: Literal["train", "test"]):
        self.split = split
        settings = SettingManager.get_settings()
        self.max_length = settings.model.max_length
        self.window_size = settings.model.window_size
        data = self.__load_data(settings)
        self.origin_length = len(data)
        self.data = self.__split_and_flatten_data(data)
        self.tokenizer = BertTokenizer.from_pretrained(settings.data.tokenizer)

    def __load_data(self, settings: Settings) -> list[dict]:
        if self.split == "train":
            data_path = settings.data.train
        elif self.split == "test":
            data_path = settings.data.test
        else:
            raise NotImplementedError
        if not data_path:
            raise ValueError(f"Dataset path is empty")
        data_path = Path(data_path)
        if not data_path.is_file():
            raise ValueError(f"Dataset path is not a file")
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # validate
        if not isinstance(data, list):
            raise ValueError(f"Data must be a list")
        if not all(isinstance(d, dict) for d in data):
            raise ValueError(f"Data item must be a dict")
        if not all("resumes" in d for d in data):
            raise ValueError(f"Data item must have 'resumes' key")
        return data

    def __split_and_flatten_data(self, data: list[dict]) -> list[tuple]:
        flatten_data = []
        for d in data:
            resumes = d["resumes"]
            for i in range(len(resumes) - self.window_size):
                history = resumes[i:i + self.window_size]
                target = resumes[i + self.window_size]
                flatten_data.append((history, target))
        return flatten_data


    def __getitem__(self, idx):
        history, target = self.data[idx]
        history: list[str]
        target: str

        # split time and resume
        times = []
        resumes = []
        for resume in history:
            time, resume = resume.split(" ", 1)
            times.append(time.strip())
            resumes.append(resume.strip())
        target = target.split(" ", 1)[1].strip()

        # extract features
        time_features = self.__extract_time_features(times)
        resume_features = self.__extract_resume_features(resumes)

        # convert to tensor
        time_features = torch.tensor(time_features, dtype=torch.float)  # (window_size, 1)
        resume_input_ids = resume_features["input_ids"]  # (window_size, seq_len)
        resume_attention_mask = resume_features["attention_mask"]  # (window_size, seq_len)
        return {
            "window_time_features": time_features,
            "window_resume_input_ids": resume_input_ids,
            "window_resume_attention_mask": resume_attention_mask,
            "target_resume": target
        }

    def __extract_time_features(self, times: list[str]) -> list[list[int]]:
        time_features = []
        for time in times:
            start, end = time.split("-", 1)
            # skip last resume
            if end.strip() in ["", "至今"]:
                continue

            start_year, start_month = map(int, re.findall(r"\d+", start))
            start_time = datetime(start_year, start_month, 1)
            end_year, end_month = map(int, re.findall(r"\d+", end))
            end_time = datetime(end_year, end_month, 1)

            # month difference as features
            months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
            time_features.append([months])
        return time_features

    def __extract_resume_features(self, resumes: list[str]):
        # (window_size, seq_len)
        resume_features = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=resumes,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return resume_features

    def __len__(self):
        return len(self.data)
