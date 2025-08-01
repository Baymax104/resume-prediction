# -*- coding: UTF-8 -*-
import json


samples: list[dict] = json.load(open("../../data/samples.json", "r", encoding="utf-8"))


def extract_targets(texts: list[str]) -> list[str]:
    targets = [text.split(" ", 1)[1].strip() for text in texts]
    return targets


resume_targets = []
for sample in samples:
    targets = extract_targets(sample["resumes"])
    resume_targets.extend(targets)

json.dump(resume_targets, open("../../data/target.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
