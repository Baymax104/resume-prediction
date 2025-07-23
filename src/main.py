# -*- coding: UTF-8 -*-
import torch.cuda
from torch import nn, optim
from tqdm import tqdm

from data import ResumeDataLoader, ResumeDataset
from model.embedding import ResumeEmbedder
from model.model import ResumePredictor
from setting import SettingManager, Settings
from utils.model import save_model, set_seed
from utils.recorder import Recorder


def train(settings: Settings):
    dataset = ResumeDataset("train", settings)
    loader = ResumeDataLoader(dataset)
    device = settings.train.device if not settings.debug and torch.cuda.is_available() else "cpu"

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
    recorder = Recorder(["loss"], mode="train", settings=settings)

    model.train()
    print('=' * 50 + f'Training in {device}' + '=' * 50)
    for e in range(1, settings.train.epochs + 1):
        train_loss = 0.
        for batch in tqdm(loader, desc=f'Training Epoch [{e}/{settings.train.epochs}]', colour='green'):
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

            loss = criterion(result_embedding, target_embedding)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(loader)
        train_metrics = {"loss": train_loss}
        recorder.add_record(train_metrics)

        if e % 5 == 0:
            recorder.print(clean=True)

    print(f"Training [Loss]: {min(recorder["loss"])}")
    recorder.plot()
    print('Saving model...')
    save_model(model, "resume-predictor", settings.log.log_dir)
    print("Done!")


if __name__ == "__main__":
    settings = SettingManager.get_settings()
    set_seed(settings.train.seed)
    train(settings)
