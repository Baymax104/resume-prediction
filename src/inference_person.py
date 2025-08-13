# -*- coding: UTF-8 -*-
from pathlib import Path
from typing import Annotated

import torch
from icecream import ic
from torchmetrics import CosineSimilarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typer import Argument, Option

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
    root = Path(__file__).parent.parent
    model_path = root / "model" / "bge-large-zh-v1.5"
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


@torch.no_grad()
def inference(
    data_path: Path,
    weight_path: Path,
    position: str,
    device: str,
    batch_size: int,
):
    dataset = InferenceDataset(data_path)
    loader = InferenceDataLoader(dataset, batch_size=batch_size)
    print(f"Data loaded, total {len(dataset)}")

    print("Embedding position...")
    # (batch_size, 1024)
    target_embedding = embedding_position(position, device)
    target_embedding = target_embedding.expand(batch_size, -1)

    print("Initializing model...")
    model = init_model(weight_path, device)
    metrics = CosineSimilarity(reduction="none").to(device)
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

    result_scores = sorted(result_scores, key=lambda x: x["score"], reverse=True)
    return result_scores


def main(
    data_path: Annotated[Path, Argument(exists=True, file_okay=True)],
    weight_path: Annotated[Path, Argument(exists=True, file_okay=True)],
    position: Annotated[str, Argument()],
    device: Annotated[str, Option()] = "cuda",
    batch_size: Annotated[int, Option()] = 64,
    seed: Annotated[int, Option()] = 200,
    limit: Annotated[int, Option()] = 10,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    result_scores = inference(
        data_path=data_path,
        weight_path=weight_path,
        position=position,
        device=device,
        batch_size=batch_size,
    )
    results = result_scores[:limit]
    ic(results)


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "samples.json"
    weight_path = Path(__file__).parent.parent / "logs/model/bge-mlp-20250803-160155.pt"
    position = "北京市朝阳区副区长"
    main(data_path, weight_path, position)
