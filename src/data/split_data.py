# -*- coding: UTF-8 -*-
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


root = Path(__file__).parent.parent.parent
data: list = json.load(open(root / "data" / "data.json", "r", encoding="utf-8"))
train, test = train_test_split(data, test_size=0.2, random_state=42)

json.dump(train, open(root / "data" / "train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(test, open(root / "data" / "test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
