# -*- coding: UTF-8 -*-
from icecream import ic

from data import ResumeDataLoader, ResumeDataset
from model.model import ResumePredictor


if __name__ == "__main__":
    dataset = ResumeDataset("train")
    loader = ResumeDataLoader(dataset)
    it = iter(loader)
    sample = next(it)
    model = ResumePredictor(window_size=2)
    output = model(sample)
    ic(output)
