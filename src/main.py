# -*- coding: UTF-8 -*-
from icecream import ic

from data import ResumeDataLoader, ResumeDataset


if __name__ == "__main__":
    dataset = ResumeDataset("train")
    loader = ResumeDataLoader(dataset)
    it = iter(loader)
    sample = next(it)
    ic(sample)
