# -*- coding: UTF-8 -*-
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


model_path = Path(__file__).parent.parent.parent / "model" / "bge-large-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_path.resolve())
model = AutoModel.from_pretrained(model_path.resolve())

texts = ["Hello", "World"]

encoded = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=texts,
    padding="max_length",
    max_length=32,
    truncation=True,
    return_tensors="pt",
)
