# -*- coding: UTF-8 -*-
from torch.utils.data import DataLoader

from data.dataset import ResumeDataset
from setting import SettingManager


class ResumeDataLoader(DataLoader):

    def __init__(self, dataset: ResumeDataset):
        settings = SettingManager.get_settings()
        data_settings = settings.data
        super().__init__(
            dataset=dataset,
            batch_size=data_settings.batch_size,
            shuffle=(dataset.split == "train"),
            num_workers=data_settings.num_workers,
            pin_memory=True
        )
