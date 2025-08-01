# -*- coding: UTF-8 -*-
from pathlib import Path

import requests
import torch
from icecream import ic
from transformers import AutoModel, AutoTokenizer


root = Path(__file__).parent.parent.parent


def inference(text: str) -> list[float]:
    model_path = root / "model" / "bge-large-zh-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_path.resolve())
    model = AutoModel.from_pretrained(model_path.resolve(), device_map="auto")
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    with torch.no_grad():
        text_emb = model(input_ids=input_ids, attention_mask=attention_mask)
    text_emb = text_emb.pooler_output.squeeze().tolist()
    return text_emb


def search(text_emb: list[float]):
    query = {
        "knn": {
            "field": "content_vector",
            "query_vector": text_emb,
            "k": 10,
            "num_candidates": 100
        },
    }

    url = "http://localhost:9200/resumes/_search"
    response = requests.post(url, json=query)
    results = response.json()["hits"]
    return results


if __name__ == "__main__":
    text = "信息科技"
    text_emb = inference(text)
    results = search(text_emb)
    ic(results)
