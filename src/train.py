# -*- coding: UTF-8 -*-
from pathlib import Path
from typing import Annotated

import torch
import typer
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import CosineSimilarity, MetricCollection
from tqdm import tqdm
from typer import Option

from components import Recorder
from data import ResumeDataLoader, ResumeDataset
from model import ResumeEmbedding, ResumePredictor
from setting import SettingManager, Settings
from utils import fix_env, load_model, log_train_settings, save_model, set_seed


def train(
    settings: Settings,
    checkpoint: int | None,
    init: Path | None,
    env: str
):
    dataset = ResumeDataset("train", settings)
    loader = ResumeDataLoader(dataset)
    device = settings.train.device
    epochs = settings.train.epochs

    model = ResumePredictor(
        pretrained_model_dir=settings.model.pretrained_model_dir,
        hidden_dim=settings.model.d_model,
        dropout=settings.model.dropout,
        output_dim=1024,
    ).to(device)

    target_embedder = ResumeEmbedding().to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=settings.train.lr,
        weight_decay=settings.train.weight_decay
    )
    criterion = nn.CosineEmbeddingLoss()
    recorder = Recorder(["loss", "cosine similarity"], mode="train", settings=settings)
    metrics = MetricCollection({
        "cosine similarity": CosineSimilarity(reduction="mean")
    }).to(device)
    start = 1

    if init and not checkpoint:
        print("Loading initialized model...")
        load_model(model, init)

    if checkpoint and not init:
        print("Loading checkpoint...")
        states = recorder.load_checkpoint(checkpoint)
        model.load_state_dict(states["model"])
        optimizer.load_state_dict(states["optimizer"])
        start = states["epoch"] + 1

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)

    model.train()
    print('=' * 20 + f' Training in {device} using {env} environment ' + '=' * 20)
    log_train_settings(settings)
    print(f"Training epoch starts from {start}")
    for e in range(start, epochs + 1):
        train_loss = 0.
        for batch in tqdm(loader, desc=f'Training Epoch [{e}/{epochs}]', colour='green'):
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
            train_loss += loss.item()
            metrics.update(result_embedding, target_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss /= len(loader)
        train_metrics = metrics.compute()
        train_metrics["loss"] = train_loss
        recorder.add_record(train_metrics)

        if e % 5 == 0:
            recorder.print(clean=True)

        if e % settings.train.checkpoint_step == 0:
            print(f"Saving checkpoint...")
            recorder.save_checkpoint(model, optimizer, e)

    print(f"Train [Loss]: {min(recorder['loss'])}")
    if not settings.debug:
        recorder.plot()
        print('Saving model...')
        save_model(model, "bge-mlp", settings.log.log_dir)
    print("Done!")


def main(
    env: Annotated[str, Option()] = "dev",
    checkpoint: Annotated[int | None, Option()] = None,
    init: Annotated[Path | None, Option(exists=True, file_okay=True)] = None,
):
    env = fix_env(env)
    if checkpoint and init:
        raise ValueError(f"Checkpoint and init are mutually exclusive")
    settings = SettingManager.get_settings(env)
    set_seed(settings.train.seed)
    train(settings, checkpoint, init, env)


if __name__ == "__main__":
    typer.run(main)
