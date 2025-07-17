# -*- coding: UTF-8 -*-
from torch.utils.data import DataLoader

from data.dataset import ResumeDataset
from setting import SettingManager


class ResumeDataLoader(DataLoader):

    def __init__(self, dataset: ResumeDataset):
        settings = SettingManager.get_settings()
        data_settings = settings.data
        debug = settings.debug
        super().__init__(
            dataset=dataset,
            batch_size=data_settings.batch_size if not debug else 1,
            shuffle=(dataset.split == "train" and not debug),
            num_workers=data_settings.num_workers if not debug else 0,
            pin_memory=(not debug)
        )
