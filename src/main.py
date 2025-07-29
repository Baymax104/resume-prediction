# -*- coding: UTF-8 -*-
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import MeanSquaredError, MetricCollection
from tqdm import tqdm

from data import ResumeDataLoader, ResumeDataset
from model import ResumeEmbedding, ResumePredictor
from setting import SettingManager, Settings
from utils import Recorder, save_model, set_seed


def train(settings: Settings):
    dataset = ResumeDataset("train", settings)
    loader = ResumeDataLoader(dataset)
    device = settings.train.device
    epochs = settings.train.epochs

    model = ResumePredictor(
        d_model=settings.model.d_model,
        num_heads=settings.model.num_heads,
        num_layers=settings.model.num_layers,
        dim_feedforward=settings.model.dim_feedforward,
        window_size=settings.model.window_size,
        dropout=settings.model.dropout,
        embedding_dim=1024,
    ).to(device)

    target_embedder = ResumeEmbedding().to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=settings.train.lr,
        weight_decay=settings.train.weight_decay
    )
    criterion = nn.CosineEmbeddingLoss()
    metrics = MetricCollection({"mse": MeanSquaredError()}).to(device)
    recorder = Recorder(["loss", "mse"], mode="train", settings=settings)

    model.train()
    print('=' * 50 + f' Training in {device} ' + '=' * 50)
    for e in range(1, epochs + 1):
        train_loss = 0.
        for batch in tqdm(loader, desc=f'Training Epoch [{e}/{epochs}]', colour='green'):
            window_resume_time = batch["window_resume_time"].to(device)
            window_resume_input_ids = batch["window_resume_input_ids"].to(device)
            window_resume_attention_mask = batch["window_resume_attention_mask"].to(device)
            target_resume_input_ids = batch["target_resume_input_ids"].to(device)
            target_resume_attention_mask = batch["target_resume_attention_mask"].to(device)

            result_embedding = model(
                input_ids=window_resume_input_ids,
                attention_mask=window_resume_attention_mask,
                resume_time=window_resume_time
            )

            target_embedding = target_embedder(
                input_ids=target_resume_input_ids,
                attention_mask=target_resume_attention_mask
            )

            target_labels = torch.ones(result_embedding.size(0), dtype=torch.float).to(device)

            result_embedding = F.normalize(result_embedding, dim=1)
            target_embedding = F.normalize(target_embedding, dim=1)

            loss = criterion(result_embedding, target_embedding, target_labels)
            train_loss += loss.item()
            metrics.update(result_embedding, target_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(loader)
        train_metrics = metrics.compute()
        train_metrics["loss"] = train_loss
        recorder.add_record(train_metrics)
        metrics.reset()

        if e % 5 == 0:
            recorder.print(clean=True)

    print(f"Training [Loss]: {min(recorder['loss'])}")
    print(f"Training [MSE]: {min(recorder['mse'])}")
    if not settings.debug:
        recorder.plot()
        print('Saving model...')
        save_model(model, "resume-predictor", settings.log.log_dir)
    print("Done!")


if __name__ == "__main__":
    settings = SettingManager.get_settings()
    set_seed(settings.train.seed)
    train(settings)
