# -*- coding: UTF-8 -*-
import json
from pathlib import Path

import requests
import torch
from requests import Session
from tqdm import tqdm


root = Path(__file__).parent.parent.parent

data = torch.load(root / "data" / "embeddings.pt")

batch_data = [data[i:i + 10] for i in range(0, len(data), 10)]


def generate_document(data: dict) -> str:
    index = {
        "index": {
            "_index": "resumes",
            "_id": data["id"],
        }
    }
    doc = {
        "content": data["text"],
        "content_vector": data["emb"].cpu().tolist()
    }
    return f"{json.dumps(index, ensure_ascii=False)}\n{json.dumps(doc, ensure_ascii=False)}\n"


def insert(session: Session, data: str):
    url = "http://localhost:9200/_bulk"
    headers = {"content-type": "application/x-ndjson"}
    session.post(url, headers=headers, data=data)


def main():
    with requests.Session() as session:
        for batch in tqdm(batch_data):
            data = []
            for item in batch:
                data.append(generate_document(item))
            data = "".join(data)
            insert(session, data)


if __name__ == "__main__":
    main()
