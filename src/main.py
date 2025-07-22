# -*- coding: UTF-8 -*-
from torch import nn, optim
from tqdm import tqdm

from data import ResumeDataLoader, ResumeDataset
from model.embedding import ResumeEmbedder
from model.model import ResumePredictor
from setting import SettingManager, Settings
from utils.model import set_seed
from utils.recorder import Recorder


def train(settings: Settings):
    dataset = ResumeDataset("train", settings)
    loader = ResumeDataLoader(dataset)
    device = settings.train.device

    model = ResumePredictor(
        d_model=settings.model.d_model,
        num_heads=settings.model.num_heads,
        num_layers=settings.model.num_layers,
        dim_feedforward=settings.model.dim_feedforward,
        window_size=settings.model.window_size,
        embedding_dim=1024,
        dropout=settings.model.dropout
    ).to(device)

    target_embedder = ResumeEmbedder().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=settings.train.lr,
        weight_decay=settings.train.weight_decay
    )
    criterion = nn.MSELoss().to(device)
    recorder = Recorder(["mse"], mode="train", settings=settings)

    print('=' * 50 + f'Training in {device}' + '=' * 50)
    for e in range(1, settings.train.epochs + 1):
        train_loss = 0.
        for batch in tqdm(loader, desc=f'Training Epoch [{e}/{settings.train.epochs}]', colour='green'):
            window_time_features = batch["window_time_features"].to(device)
            window_resume_input_ids = batch["window_resume_input_ids"].to(device)
            window_resume_attention_mask = batch["window_resume_attention_mask"].to(device)
            target_resume_input_ids = batch["target_resume_input_ids"].to(device)
            target_resume_attention_mask = batch["target_resume_attention_mask"].to(device)

            result_embedding = model(
                input_ids=window_resume_input_ids,
                attention_mask=window_resume_attention_mask,
                time_features=window_time_features
            )

            target_embedding = target_embedder(
                input_ids=target_resume_input_ids,
                attention_mask=target_resume_attention_mask
            )

            loss = criterion(result_embedding, target_embedding)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(loader)
        train_metrics = {"mse": train_loss}
        recorder.add_record(train_metrics)

        if e % 5 == 0:
            recorder.print(clean=True)

    print(f'Training [Loss]: {min(recorder["mse"])}')


if __name__ == "__main__":
    settings = SettingManager.get_settings()
    set_seed(settings.train.seed)
    train(settings)
