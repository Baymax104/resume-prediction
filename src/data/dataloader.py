# -*- coding: UTF-8 -*-
from torch.utils.data import DataLoader

from data.dataset import ResumeDataset


class ResumeDataLoader(DataLoader):

    def __init__(self, dataset: ResumeDataset):
        settings = dataset.settings
        data_settings = settings.data
        debug = settings.debug
        super().__init__(
            dataset=dataset,
            batch_size=data_settings.batch_size,
            shuffle=(dataset.split == "train" and not debug),
            num_workers=data_settings.num_workers if not debug else 0,
            pin_memory=(not debug)
        )
