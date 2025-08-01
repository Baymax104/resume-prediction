# -*- coding: UTF-8 -*-
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


root = Path(__file__).parent.parent.parent

model_path = root / "model" / "bge-large-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_path.resolve())
model = AutoModel.from_pretrained(model_path.resolve(), device_map="auto")

texts: list[str] = json.load(open(root / "data" / "target.json", "r", encoding="utf-8"))

batch_size = 512
batch_texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


def encode_text(texts: list[str]) -> tuple:
    encoded = tokenizer(
        texts,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    return input_ids, attention_mask


def inference(input_ids, attention_mask):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    return output.pooler_output


def main():
    embeddings = []
    for batch_text in tqdm(batch_texts):
        encoded = encode_text(batch_text)
        output = inference(*encoded)
        tensors = torch.unbind(output, dim=0)
        text_emb = [
            {
                "text": text,
                "emb": tensors[i]
            }
            for i, text in enumerate(batch_text)
        ]
        embeddings.extend(text_emb)
    print("Saving...")
    torch.save(embeddings, root / "data" / "embeddings.pt")


if __name__ == "__main__":
    main()
