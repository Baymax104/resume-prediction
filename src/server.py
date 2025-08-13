# -*- coding: UTF-8 -*-
import asyncio
from pathlib import Path
from typing import Annotated

import torch
import uvicorn
from fastapi import Body, FastAPI
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


root = Path(__file__).parent.parent
data_path = root / "data" / "samples.json"
weight_path = root / "logs/model/bge-mlp-20250803-160155.pt"
batch_size = 4
device = "cuda"
dataset = InferenceDataset(data_path)
model = init_model(weight_path, device)


@torch.no_grad()
def inference(position: str, limit: int):
    print("Embedding position...")
    # (batch_size, 1024)
    target_embedding = embedding_position(position, device)
    target_embedding = target_embedding.expand(batch_size, -1)

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

        similarity = metrics(result_embedding, target_embedding)
        similarity = similarity.tolist()
        for i in range(batch_size):
            result_scores.append({
                "name": items[i]["name"],
                "score": similarity[i],
                "resume": items[i]["resumes"]
            })
        break

    result_scores = sorted(result_scores, key=lambda x: x["score"], reverse=True)
    result_scores = result_scores[:limit]
    return result_scores


app = FastAPI()


@app.post("/search")
async def search(
    text: Annotated[str, Body()],
    limit: Annotated[int, Body()] = 10
):
    logger.debug(f"{text=}, {limit=}")
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, inference, text, limit)
    logger.debug(f"{results=}")
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
