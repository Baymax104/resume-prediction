# -*- coding: UTF-8 -*-
from icecream import ic

from data import ResumeDataset


if __name__ == "__main__":
    dataset = ResumeDataset("train")
    sample = dataset[0]
    ic(sample)
