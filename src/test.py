# -*- coding: UTF-8 -*-
from pathlib import Path
from typing import Annotated

import torch
import typer
from torch import nn
from torchmetrics import CosineSimilarity, MetricCollection
from tqdm import tqdm
from typer import Argument, Option

from components import Recorder
from data import ResumeDataLoader, ResumeDataset
from model import ResumeEmbedding, ResumePredictor
from setting import SettingManager, Settings
from utils import fix_env, load_model, set_seed


@torch.no_grad()
def test(settings: Settings, model_path: Path, env: str):
    dataset = ResumeDataset("test", settings)
    loader = ResumeDataLoader(dataset)
    device = settings.train.device

    model = ResumePredictor(
        pretrained_model_dir=settings.model.pretrained_model_dir,
        hidden_dim=settings.model.d_model,
        dropout=settings.model.dropout,
        output_dim=1024,
    ).to(device)

    target_embedder = ResumeEmbedding().to(device)

    load_model(model, model_path)
    model.eval()

    criterion = nn.CosineEmbeddingLoss()
    recorder = Recorder(["loss", "cosine similarity"], mode="test", settings=settings)
    metrics = MetricCollection({
        "cosine similarity": CosineSimilarity(reduction="mean")
    }).to(device)

    print('=' * 20 + f' Testing in {device} using {env} environment' + '=' * 20)
    test_loss = 0.
    for batch in tqdm(loader, desc=f"Testing", colour="green"):
        window_resume_input_ids = batch["window_resume_input_ids"].to(device)
        window_resume_attention_mask = batch["window_resume_attention_mask"].to(device)
        window_resume_token_type_ids = batch["window_resume_token_type_ids"].to(device)
        target_resume_input_ids = batch["target_resume_input_ids"].to(device)
        target_resume_attention_mask = batch["target_resume_attention_mask"].to(device)

        result_embedding = model(
            input_ids=window_resume_input_ids,
            attention_mask=window_resume_attention_mask,
            token_type_ids=window_resume_token_type_ids,
        )

        target_embedding = target_embedder(
            input_ids=target_resume_input_ids,
            attention_mask=target_resume_attention_mask
        )

        target_labels = torch.ones(result_embedding.size(0), dtype=torch.float).to(device)

        loss = criterion(result_embedding, target_embedding, target_labels)
        test_loss += loss.item()
        metrics.update(result_embedding, target_embedding)

    test_loss /= len(loader)
    test_metrics = metrics.compute()
    test_metrics["loss"] = test_loss
    recorder.add_record(test_metrics)
    recorder.print()


def main(
    model_path: Annotated[Path, Argument(exists=True, file_okay=True)],
    env: Annotated[str, Option()] = "dev",
):
    env = fix_env(env)
    settings = SettingManager.get_settings(env)
    set_seed(settings.train.seed)
    test(settings, model_path, env)


if __name__ == "__main__":
    typer.run(main)
