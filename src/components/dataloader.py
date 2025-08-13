# -*- coding: UTF-8 -*-
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from components.dataset import InferenceDataset, ResumeDataset


class ResumeDataLoader(DataLoader):

    def __init__(self, dataset: ResumeDataset):
        settings = dataset.settings
        data_settings = settings.data
        debug = settings.debug
        super().__init__(
            dataset=dataset,
            batch_size=data_settings.batch_size,
            shuffle=(dataset.split == "train" and not debug),
            num_workers=data_settings.num_workers,
            pin_memory=(not debug)
        )


def collate_fn(batch):
    items = [sample["item"] for sample in batch]
    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]
    token_type_ids = [sample["token_type_ids"] for sample in batch]
    return {
        "item": items,
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "token_type_ids": torch.stack(token_type_ids)
    }


class InferenceDataLoader(DataLoader):

    def __init__(self, dataset: InferenceDataset, batch_size: int = 64):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
