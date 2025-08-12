# -*- coding: UTF-8 -*-
import re
from pathlib import Path
from typing import Annotated

import requests
import torch
from icecream import ic
from transformers import AutoTokenizer
from typer import Argument

from model import ResumePredictor


def preprocessing(resume: str) -> list[str]:
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


def load_model(model, model_path, device="cuda"):
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise ValueError(f'Model {str(model_path)} does not exist.')
    state_dict = torch.load(model_path, map_location=device)
    state_dict = state_dict['model'] if 'model' in state_dict else state_dict
    model.load_state_dict(state_dict)


@torch.no_grad()
def inference(model_path: Path, resume: str) -> list[float]:
    print("Initializing model...")
    device = "cuda"
    root = Path(__file__).parent.parent
    model_dir = root / "model"
    tokenizer = AutoTokenizer.from_pretrained((model_dir / "bge-large-zh-v1.5").resolve())

    model = ResumePredictor(
        pretrained_model_dir=model_dir,
        hidden_dim=1024,
        dropout=0.1,
        output_dim=1024,
    ).to(device)

    load_model(model, model_path, device=device)
    model.eval()

    print("Preprocessing data...")
    resumes = preprocessing(resume)
    sample = encode(tokenizer, resumes)
    window_resume_input_ids = sample["window_resume_input_ids"].to(device)
    window_resume_attention_mask = sample["window_resume_attention_mask"].to(device)
    window_resume_token_type_ids = sample["window_resume_token_type_ids"].to(device)

    result_embedding = model(
        input_ids=window_resume_input_ids,
        attention_mask=window_resume_attention_mask,
        token_type_ids=window_resume_token_type_ids,
    )

    result_embedding = result_embedding.detach().cpu().numpy()
    result_embedding = result_embedding.squeeze().tolist()
    return result_embedding


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
    results = response.json()["hits"]["hits"]
    results = [{"score": r["_score"], "content": r["_source"]["content"]} for r in results]
    return results


def main(
    model_path: Annotated[Path, Argument(exists=True, file_okay=True)],
    data: Annotated[str, Argument()]
):
    torch.manual_seed(200)
    torch.cuda.manual_seed_all(200)
    embedding = inference(model_path, data)
    results = search(embedding)
    ic(results)


if __name__ == "__main__":
    resume = """
    2010.07-2015.03 上海市徐汇区发改委科员
    2015.04-2018.06 上海市徐汇区发改委副主任
    2018.07-2021.11 上海市虹口区政府办公室主任
    2021.12-        上海市虹口区副区长
    """
    model_path = Path(__file__).parent.parent / "logs/model/bge-mlp-20250803-160155.pt"
    main(model_path, resume)
