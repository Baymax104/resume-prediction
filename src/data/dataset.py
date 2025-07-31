# -*- coding: UTF-8 -*-
import json
import re
from datetime import datetime
from typing import Literal

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from setting import Settings


class ResumeDataset(Dataset):

    def __init__(self, split: Literal["train", "test"], settings: Settings):
        self.split = split
        self.settings = settings
        self.batch_size = settings.data.batch_size
        self.max_length = settings.model.max_length
        self.window_size = settings.model.window_size

        data = self.__load_data(settings)
        self.origin_length = len(data)
        self.data = self.__split_and_flatten_data(data)
        if settings.debug:
            self.data = self.data[:self.batch_size]

        model_dir = settings.model.pretrained_model_dir
        bert_path = model_dir / "bert-base-chinese"
        bge_path = model_dir / "bge-large-zh-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path.resolve(), use_fast=True)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(bge_path.resolve(), use_fast=True)

    def __load_data(self, settings: Settings) -> list[dict]:
        if self.split == "train":
            data_path = settings.data.train
        elif self.split == "test":
            data_path = settings.data.test
        else:
            raise NotImplementedError
        if not data_path or not data_path.is_file():
            raise ValueError(f"Dataset path is empty")
        data = json.load(data_path.open("r", encoding="utf-8"))
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

        # # split time and resume
        # times = []
        # resumes = []
        # for resume in history:
        #     time, resume = resume.split(" ", 1)
        #     times.append(time.strip())
        #     resumes.append(resume.strip())
        # target = target.split(" ", 1)[1].strip()
        #
        # # extract features
        # time_features = self.__extract_time_features(times)
        # resume_features = self.__extract_resume_features(resumes)
        # target_feature = self.__extract_target_features(target)
        resume_features = self.__extract_resume_features(history)
        target_feature = self.__extract_target_features(target)

        # convert to tensor
        # resume_time = torch.tensor(time_features, dtype=torch.float)  # (window_size, 1)
        resume_input_ids = resume_features["input_ids"].squeeze()  # (seq_len,)
        resume_attention_mask = resume_features["attention_mask"].squeeze()  # (seq_len,)
        resume_token_type_ids = resume_features["token_type_ids"].squeeze()  # (seq_len,)
        target_input_ids = target_feature["input_ids"].squeeze()  # (seq_len,)
        target_attention_mask = target_feature["attention_mask"].squeeze()  # (seq_len,)
        return {
            # "window_resume_time": resume_time,
            "window_resume_input_ids": resume_input_ids,
            "window_resume_attention_mask": resume_attention_mask,
            "window_resume_token_type_ids": resume_token_type_ids,
            "target_resume_input_ids": target_input_ids,
            "target_resume_attention_mask": target_attention_mask,
        }

    def __extract_time_features(self, times: list[str]) -> list[list[int]]:
        time_features = []
        for time in times:
            start, end = time.split("-", 1)
            # current date
            if end.strip() in ["", "至今"]:
                end = datetime.now().strftime("%Y.%m")

            start_year, start_month = map(int, re.findall(r"\d+", start))
            start_time = datetime(start_year, start_month, 1)
            end_year, end_month = map(int, re.findall(r"\d+", end))
            end_time = datetime(end_year, end_month, 1)

            # month difference as features
            months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
            time_features.append([months])
        return time_features

    def __extract_resume_features(self, resumes: list[str]):
        # combine sentences in a window
        # (1, seq_len)
        resume_features = self.tokenizer(
            [resumes],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        return resume_features

    def __extract_target_features(self, target_resume: str):
        # (1, seq_len)
        target_feature = self.embedding_tokenizer(
            text=target_resume,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return target_feature

    def __len__(self):
        return len(self.data)
