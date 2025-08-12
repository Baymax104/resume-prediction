# -*- coding: UTF-8 -*-
import json
from pathlib import Path
from typing import Annotated

import torch
from icecream import ic
from torchmetrics import CosineSimilarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typer import Argument

from model import ResumePredictor


def init_tokenizer(root: Path):
    model_dir = root / "model"
    tokenizer = AutoTokenizer.from_pretrained((model_dir / "bge-large-zh-v1.5").resolve())
    return tokenizer


def init_model(root: Path, weight_path: Path, device="cuda"):
    model_dir = root / "model"
    model = ResumePredictor(
        pretrained_model_dir=model_dir,
        hidden_dim=1024,
        dropout=0.1,
        output_dim=1024,
    ).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict['model'] if 'model' in state_dict else state_dict
    model.load_state_dict(state_dict)
    return model


def load_data(path: Path) -> list[dict]:
    data: list[dict] = json.load(path.open("r", encoding="utf-8"))
    return data


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


def encode(tokenizer, resumes) -> dict[str, torch.Tensor]:
    history = resumes[-2:]
    resume_features = tokenizer(
        [history],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    resume_input_ids = resume_features["input_ids"]  # (1, seq_len)
    resume_attention_mask = resume_features["attention_mask"]  # (1, seq_len)
    resume_token_type_ids = resume_features["token_type_ids"]  # (1, seq_len)
    return {
        "window_resume_input_ids": resume_input_ids,
        "window_resume_attention_mask": resume_attention_mask,
        "window_resume_token_type_ids": resume_token_type_ids,
    }


@torch.no_grad()
def inference(data_path: Path, weight_path: Path, position: str) -> list[float]:
    data = load_data(data_path)
    print(f"Data loaded, total {len(data)}")
    print("Initializing model...")
    device = "cuda"
    root = Path(__file__).parent.parent
    tokenizer = init_tokenizer(root)
    model = init_model(root, weight_path, device)
    metrics = CosineSimilarity(reduction="none").to(device)
    result_scores = []

    model.eval()
    print("Embedding position...")
    # (1, 1024)
    target_embedding = embedding_position(position, device)
    for sample in tqdm(data, desc="Predicting person jobs"):
        resumes = sample["resumes"]
        encoded_resumes = encode(tokenizer, resumes)
        window_resume_input_ids = encoded_resumes["window_resume_input_ids"].to(device)
        window_resume_attention_mask = encoded_resumes["window_resume_attention_mask"].to(device)
        window_resume_token_type_ids = encoded_resumes["window_resume_token_type_ids"].to(device)

        # (1, 1024)
        result_embedding = model(
            input_ids=window_resume_input_ids,
            attention_mask=window_resume_attention_mask,
            token_type_ids=window_resume_token_type_ids,
        )

        similarity = metrics(result_embedding, target_embedding)
        similarity = similarity.item()
        result = {
            "id": sample["name"],
            "score": similarity,
            "resume": sample["resumes"]
        }
        result_scores.append(result)

    result_scores = sorted(result_scores, key=lambda x: x["score"], reverse=True)
    return result_scores


def main(
    data_path: Annotated[Path, Argument(exists=True, file_okay=True)],
    weight_path: Annotated[Path, Argument(exists=True, file_okay=True)],
    position: Annotated[str, Argument()],
):
    torch.manual_seed(200)
    torch.cuda.manual_seed_all(200)
    result_scores = inference(data_path, weight_path, position)
    results = result_scores[:10]
    ic(results)


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "samples.json"
    weight_path = Path(__file__).parent.parent / "logs/model/bge-mlp-20250803-160155.pt"
    position = "北京市朝阳区副区长"
    main(data_path, weight_path, position)
