# -*- coding: UTF-8 -*-
import asyncio
import re
from pathlib import Path
from typing import Annotated

import requests
import torch
import uvicorn
from fastapi import Body, FastAPI
from icecream import ic
from loguru import logger
from torchmetrics import CosineSimilarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from components import InferenceDataLoader, InferenceDataset
from model import ResumePredictor


def init_model(weight_path: Path, device="cuda"):
    root = Path(__file__).parent.parent
    model_dir = root / "model"
    model = ResumePredictor(
        pretrained_model_dir=model_dir,
        hidden_dim=1024,
        output_dim=1024,
    ).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict['model'] if 'model' in state_dict else state_dict
    model.load_state_dict(state_dict)
    return model


def embedding_position(position, device="cuda"):
    model_path = Path(__file__).parent.parent / "model" / "bge-large-zh-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_path.resolve())
    model = AutoModel.from_pretrained(model_path.resolve(), device_map="auto")
    encoded = tokenizer(position, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        text_emb = model(input_ids=input_ids, attention_mask=attention_mask)
    # (1, 1024)
    text_emb = text_emb.pooler_output
    return text_emb


def encode_resumes(resumes) -> dict[str, torch.Tensor]:
    model_path = Path(__file__).parent.parent / "model" / "bge-large-zh-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_path.resolve())
    resume_features = tokenizer(
        [resumes],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    input_ids = resume_features["input_ids"]  # (1, seq_len)
    attention_mask = resume_features["attention_mask"]  # (1, seq_len)
    token_type_ids = resume_features["token_type_ids"]  # (1, seq_len)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def split_resume(resume: str) -> list[str]:
    pattern = re.compile(r"(?=\d{4}.\d{2}-)")
    parts = re.split(pattern, resume)
    parts = [part.strip() for part in parts if part.strip()]
    parts = [part.replace("\r", "").replace("\n", "").replace("/r", "").replace("/n", "") for part in parts]
    resumes = []
    for part in parts:
        if match := re.match(r"(\d{4}.\d{2}-\S*)(.+)", part):
            date = match.group(1).strip()
            text = match.group(2).strip()
            text = text.replace(" ", "")
            r = f"{date} {text}"
            resumes.append(r)
    return resumes


root = Path(__file__).parent.parent
data_path = root / "data" / "samples.json"
weight_path = root / "logs/model/bge-mlp-20250803-160155.pt"
batch_size = 4
device = "cuda"
dataset = InferenceDataset(data_path)
model = init_model(weight_path, device)
es_index = "resumes"


@torch.no_grad()
def inference_persons(position: str, limit: int):
    print("Embedding position...")
    # (1, 1024)
    position_embedding = embedding_position(position, device)

    metrics = CosineSimilarity(reduction="none").to(device)
    loader = InferenceDataLoader(dataset, batch_size=batch_size)
    result_scores = []
    model.eval()
    for batch in tqdm(loader, desc="Inferencing"):
        items = batch["item"]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        # (batch_size, 1024)
        result_embedding = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        target_embedding = position_embedding.expand(result_embedding.size(0), -1)

        similarity = metrics(result_embedding, target_embedding)
        similarity = similarity.tolist()
        for i in range(result_embedding.size(0)):
            result_scores.append({
                "name": items[i]["name"],
                "score": similarity[i],
                "resume": items[i]["resumes"]
            })
        break

    result_scores = sorted(result_scores, key=lambda x: x["score"], reverse=True)
    result_scores = result_scores[:limit]
    return result_scores


def search(text_emb: list[float], limit: int):
    query = {
        "knn": {
            "field": "content_vector",
            "query_vector": text_emb,
            "k": limit,
            "num_candidates": 100
        },
    }

    url = f"http://localhost:9200/{es_index}/_search"
    response = requests.post(url, json=query)
    results = response.json()["hits"]["hits"]
    results = [
        {
            "id": r["_id"],
            "score": r["_score"],
            "content": r["_source"]["content"]
        }
        for r in results
    ]
    return results


@torch.no_grad()
def inference_jobs(resume: str, limit: int):
    print("Preprocessing resume...")
    resumes = split_resume(resume)
    if len(resumes) < 2:
        raise Exception("Not enough resumes, need at least 2")
    resumes = resumes[-2:]
    model.eval()
    encoded_resumes = encode_resumes(resumes)
    input_ids = encoded_resumes["input_ids"].to(device)
    attention_mask = encoded_resumes["attention_mask"].to(device)
    token_type_ids = encoded_resumes["token_type_ids"].to(device)

    result_embedding = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    result_embedding = result_embedding.squeeze().tolist()
    results = search(result_embedding, limit)
    return results


app = FastAPI()


@app.post("/search-persons")
async def search_persons(
    text: Annotated[str, Body()],
    limit: Annotated[int, Body()] = 10
):
    logger.debug(f"{text=}, {limit=}")
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, inference_persons, text, limit)
    ic(results)
    return results


@app.post("/search-jobs")
async def search_jobs(
    resume: Annotated[str, Body()],
    limit: Annotated[int, Body()] = 10
):
    logger.debug(f"{resume=}, {limit=}")
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, inference_jobs, resume, limit)
    ic(results)
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
